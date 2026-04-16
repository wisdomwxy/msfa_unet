"""验证用 UNet 推理默认配置。"""
import os

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# 权重：请将 88.64unetFastV2.pth 放在 weights/（见 weights/README.txt）
DEFAULT_MODEL_PATH = os.path.join(_PKG_DIR, "weights", "88.64unetFastV2.pth")
DEFAULT_OUT_DIR = os.path.join(_PKG_DIR, "out")

num_classes = 6
backbone = "mobilenetv3_small"
input_shape = [512, 512]
cuda = True

COLOR_MAP = {
    0: (0, 0, 0),
    1: (128, 0, 0),
    2: (0, 128, 0),
    3: (0, 0, 128),
    4: (128, 128, 0),
    5: (128, 0, 128),
}
