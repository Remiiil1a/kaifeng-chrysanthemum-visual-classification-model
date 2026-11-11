"""项目公共工具：路径解析、类别清单、图片枚举、模型指纹。

所有脚本都从这里取路径，避免各自硬编码盘符或绝对路径。
本模块只依赖标准库，可以在装 torch 之前先跑通。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def setup_console() -> None:
    """把 stdout/stderr 统一成 UTF-8。

    Windows 控制台默认 GBK（chcp 936）。中文输出重定向到 train_log.txt 时
    会写成 GBK 字节，别人用 UTF-8 打开就是乱码。统一成 UTF-8 后，控制台
    显示与日志文件编码一致，换机复现时也不会出现"日志里有乱码"。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


setup_console()

# code/ 的上一级就是项目根目录，因此整个项目可以整体搬走而不改代码
ROOT = Path(__file__).resolve().parent.parent

# 数据集根目录：任务书 6.1 节要求 train/val/test 直接位于根目录下
DATASET_ROOT = ROOT
SPLITS = ("train", "val", "test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# 训练产物路径，与 train.py 的 CONFIG 保持一致
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
RUN_NAME = "chrys_yolo26n"
RUN_DIR = RUNS_DIR / RUN_NAME
WEIGHTS_DIR = RUN_DIR / "weights"
BEST_WEIGHTS = WEIGHTS_DIR / "best.pt"
LAST_WEIGHTS = WEIGHTS_DIR / "last.pt"

# 元数据表与原始照片库
LABELS_CSV = ROOT / "labels.csv"
RAW_DIR = ROOT / "_raw"

# 任务书 8.1 节规定的字段顺序，build_labels.py 与 check_dataset.py 共用
LABEL_FIELDS = [
    "filename",
    "split",
    "category",
    "plant_id",
    "latitude",
    "longitude",
    "altitude",
    "datetime",
    "device",
]


def setup_env() -> None:
    """把 Ultralytics 的配置与缓存锁在项目内。

    必须在 `import ultralytics` 之前调用，否则框架已经读过默认目录了。
    任务书 10.5 节：多组共用一台电脑时，默认目录会互相覆盖。
    """
    config_dir = ROOT / ".yolo_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("OMP_NUM_THREADS", "4")


def list_classes(split: str = "train") -> list[str]:
    """返回某个子集下的类别文件夹名，按字母序排列。

    字母序即 YOLO 的类别索引顺序（任务书 11.2 节），所以两位序号保证了
    索引稳定。永远不要硬编码索引，一律用 model.names 反查。
    """
    split_dir = DATASET_ROOT / split
    if not split_dir.is_dir():
        return []
    return sorted(d.name for d in split_dir.iterdir() if d.is_dir())


def iter_images(directory: Path) -> list[Path]:
    """列出目录下（不递归）的图片文件，按文件名排序。"""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def count_images(directory: Path) -> int:
    return len(iter_images(directory))


def model_fingerprint(config: dict) -> tuple[str, str]:
    """按任务书 11.6 节计算模型指纹。

    返回 (8 位哈希, 排序后的配置 JSON)。配置项排序后再哈希，保证同样的
    配置无论字典顺序如何都得到同一个指纹。
    """
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:8].upper()
    return digest, payload


def rel(path: Path) -> str:
    """把绝对路径转成相对项目根目录的写法，用于打印，避免日志里出现盘符。"""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)
