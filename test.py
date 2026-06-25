import sys

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
else:
    print("\nGPU не используется. Для обучения на видеокарте установите PyTorch с CUDA:")
    print("  pip uninstall torch torchvision -y")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    sys.exit(1)

import ultralytics

print(f"Ultralytics version: {ultralytics.__version__}")

import cv2

print(f"OpenCV version: {cv2.__version__}")
