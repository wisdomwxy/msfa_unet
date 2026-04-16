"""
图像 -> 语义分割类别图 (H,W) uint8，值域 0..num_classes-1。
预处理与 genPhotos.process_image + predict 一致。
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from nets.unetFastV2 import Unet
from utils.utils import resize_image
import config as cfg


def load_model(model_path=None, device=None, verbose=True):
    model_path = model_path or cfg.DEFAULT_MODEL_PATH
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() and cfg.cuda else "cpu"
    )
    model = Unet(num_classes=cfg.num_classes, backbone=cfg.backbone).eval()
    if verbose:
        print("Loading weights into state dict...")
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and np.shape(model_dict[k]) == np.shape(v)
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)
    if verbose:
        print("Finished loading!")
    return model, device


def predict_logits_to_class_map(image_pil, model, device, input_shape):
    """letterbox + softmax + 裁灰边 + 双线性还原原图尺寸 + argmax。"""
    image_pil = image_pil.convert("RGB")
    orig_w, orig_h = image_pil.size
    h, w = input_shape
    resized, nw, nh = resize_image(image_pil, (w, h))
    image_data = np.expand_dims(
        np.transpose(np.array(resized, np.float32) / 255.0, (2, 0, 1)), 0
    )
    with torch.no_grad():
        images = torch.from_numpy(image_data).to(device)
        pr = model(images)[0]
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        pr = pr[
            (h - nh) // 2 : (h - nh) // 2 + nh,
            (w - nw) // 2 : (w - nw) // 2 + nw,
        ]
        pr = cv2.resize(pr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)
    return pr.astype(np.uint8)


def infer_class_map_from_path(image_path, model=None, device=None, model_path=None):
    own_model = model is None
    if own_model:
        model, device = load_model(model_path=model_path, device=device)
    image = Image.open(image_path)
    return predict_logits_to_class_map(image, model, device, cfg.input_shape)


def infer_class_map_from_bgr(image_bgr, model, device):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return predict_logits_to_class_map(pil, model, device, cfg.input_shape)
