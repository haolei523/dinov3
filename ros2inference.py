# ros2 
import rclpy
from rclpy.node import Node
from rclpy import qos
from rclpy.executors import MultiThreadedExecutor
# from sensor_msgs.msg import Image
import sensor_msgs
from cv_bridge import CvBridge

import colorsys
import os
from functools import partial
from typing import List, Tuple
from easydict import EasyDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as Fv

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, make_normalize_transform
from dinov3.eval.segmentation.inference import make_inference
from dinov3.eval.segmentation.models import BackboneLayersSet, build_segmentation_decoder
from dinov3.eval.setup import ModelConfig, load_model_and_context
from dinov3.run.init import job_context

class Dinov3Inference(Node):

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",          # 单进程 CPU/GPU 都可用，不需要 NCCL
            init_method="tcp://127.0.0.1:29500",
            world_size=1,
            rank=0
        )

    def __init__(self):
        super().__init__('dinov3_inference_node')

        args = EasyDict()
        args['backbone_weights'] = 'src/dinov3/dinov3_rosnode/pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
        args['head_ckpt'] = 'src/dinov3/dinov3_rosnode/pretrained/model_final.pth'
        args['config_file'] = 'src/dinov3/dinov3_rosnode/dinov3/configs/dinov3_vits16.yaml'
        args['num_classes'] = 6
        args['backbone_out_layers'] = 'FOUR_EVEN_INTERVALS'
        args['mode'] = 'slide'  # 'slide' or 'whole'
        args['img_size'] = 512
        args['crop_size'] = 512
        args['stride'] = 341
        args['model_dtype'] = 'float32' # 'float32' or 'bfloat16'
        args['output_dir'] = './output'
        os.makedirs(args.output_dir, exist_ok=True)

        args['device'] = torch.device('cuda')
        self.args = args
        autocast_dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float32

        self.create_subscription(sensor_msgs.msg.Image, '/zw040201/lq_camera', self.image_callback, qos_profile=qos.qos_profile_sensor_data)
        self.mask_publisher = self.create_publisher(sensor_msgs.msg.Image, 'mask_image', 10)
        self.overlay_publisher = self.create_publisher(sensor_msgs.msg.Image, 'overlay_image', 10)
        self.cv_bridge = CvBridge()

        backbone = self.build_backbone(args)

        self.seg_model = build_segmentation_decoder(
            backbone,
            BackboneLayersSet[args.backbone_out_layers],
            "segformer",
            num_classes=args.num_classes,
            autocast_dtype=autocast_dtype,
            dropout=0.0,
        ).to(args.device)

        head_sd = self.load_head_state_dict(args.head_ckpt)
        missing, unexpected = self.seg_model.load_state_dict(head_sd, strict=False)
        self.seg_model.eval()

        self.normalize = make_normalize_transform(
            mean=[m * 255 for m in IMAGENET_DEFAULT_MEAN],
            std=[s * 255 for s in IMAGENET_DEFAULT_STD],
        )

        self.palette = self.make_palette(args.num_classes)
        self.softmax = partial(F.softmax, dim=1)

    def build_backbone(self, args) -> torch.nn.Module:
        """按训练时的方式加载 backbone（离线：本地 hub 名 + 本地权重；或 config_file + 本地权重）。"""
        if args.config_file is not None:
            model_config = ModelConfig(config_file=args.config_file, pretrained_weights=args.backbone_weights)
        elif args.backbone_name is not None:
            model_config = ModelConfig(dino_hub=args.backbone_name, pretrained_weights=args.backbone_weights)
        else:
            raise ValueError("必须提供 --backbone-name 或 --config-file 之一")
        backbone, _ = load_model_and_context(model_config, output_dir=args.output_dir)
        return backbone

    def load_head_state_dict(self, path: str):
        """读取 model_final.pth 里的 head 权重（兼容 {'model': ...} 与裸 state_dict）。"""
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # 文件里若含优化器状态等非纯张量对象，weights_only 可能失败，这里回退一次
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        return ckpt

    def make_palette(self, num_classes: int) -> np.ndarray:
        """为每个类别生成确定性颜色；类别 0 视为背景（黑色）。"""
        palette = np.zeros((max(num_classes, 1), 3), dtype=np.uint8)
        for i in range(1, num_classes):
            hue = (i * 0.61803398875) % 1.0  # 黄金角，颜色分散
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
            palette[i] = (int(r * 255), int(g * 255), int(b * 255))
        return palette

    def preprocess(self, pil_img: Image.Image, img_size: int, normalize) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """复刻 make_segmentation_eval_transforms(inference_mode='slide') 的图像预处理。

        返回：([1,C,h,w] 归一化张量, 原图 (H, W))。h/w 是短边缩放到 img_size 后的尺寸。
        """
        orig_w, orig_h = pil_img.size  # PIL.size = (W, H)
        # 短边缩放到 img_size，保持长宽比（与 FixedSideResize._resize 一致）
        if orig_h > orig_w:
            new_w = img_size
            new_h = int(img_size * orig_h / orig_w + 0.5)
        else:
            new_h = img_size
            new_w = int(img_size * orig_w / orig_h + 0.5)
        img = T.Resize(size=(new_h, new_w), interpolation=T.InterpolationMode.BILINEAR)(pil_img)
        x = Fv.pil_to_tensor(img).float()  # [C,H,W]，0-255
        x = normalize(x)  # (x - mean*255) / (std*255)
        return x.unsqueeze(0), (orig_h, orig_w)

    def image_callback(self, msg):
        cv_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        pil_img = Image.fromarray(cv_img)
        x, (orig_h, orig_w) = self.preprocess(pil_img, self.args.img_size, self.normalize)
        x = x.to(self.args.device)
        with torch.inference_mode():
            pred = make_inference(
                x,
                self.seg_model,
                inference_mode=self.args.mode,
                decoder_head_type="linear",
                rescale_to=(orig_h, orig_w),
                n_output_channels=self.args.num_classes,
                crop_size=(self.args.crop_size, self.args.crop_size),
                stride=(self.args.stride, self.args.stride),
                output_activation=self.softmax,
            )

        mask = pred.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
        color = self.palette[mask]
        overlay = (0.5 * cv_img + 0.5 * color).astype(np.uint8)

        self.mask_publisher.publish(self.cv_bridge.cv2_to_imgmsg(color.astype(np.uint8), encoding='bgr8'))
        self.overlay_publisher.publish(self.cv_bridge.cv2_to_imgmsg(overlay.astype(np.uint8), encoding='bgr8'))

def main():
    rclpy.init(args=None)

    assert torch.cuda.is_available(), 'Need GPU!'

    di = Dinov3Inference()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(di)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        di.destroy_node()

if __name__ == '__main__':
    main()