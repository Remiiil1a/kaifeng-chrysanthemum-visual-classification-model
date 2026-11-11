"""环境自检脚本，运行后一次性输出所有关键信息。

用法：
    python code/check_env.py

任务书附录 A.2：把输出截图放进 README 第三部分，作为环境配置的证据。
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, rel  # noqa: E402


def main() -> None:
    print("=" * 56)
    print("Python 版本 :", sys.version.split()[0])
    print("解释器路径  :", sys.executable)
    print("操作系统    :", platform.platform())
    print("项目根目录  :", rel(ROOT))
    print("=" * 56)

    try:
        import torch

        print("PyTorch     :", torch.__version__)
        print("CUDA 可用   :", torch.cuda.is_available())
        print("CUDA 版本   :", torch.version.cuda)
        if torch.cuda.is_available():
            print("显卡        :", torch.cuda.get_device_name(0))
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"显存        : {mem:.1f} GB")
    except ImportError:
        print("PyTorch     : 未安装")

    try:
        import torchvision

        print("torchvision :", torchvision.__version__)
    except ImportError:
        print("torchvision : 未安装")

    try:
        import ultralytics

        print("Ultralytics :", ultralytics.__version__)
    except ImportError:
        print("Ultralytics : 未安装")

    try:
        import PIL

        print("Pillow      :", PIL.__version__)
    except ImportError:
        print("Pillow      : 未安装")

    print("-" * 56)
    for var in ("YOLO_CONFIG_DIR", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"):
        print(f"{var:<20}: {os.environ.get(var, '(未设置)')}")
    print("=" * 56)


if __name__ == "__main__":
    main()
