"""YOLO26 分类模型训练脚本。

用法：
    python code/train.py                # 按 CONFIG 正式训练
    python code/train.py --epochs 5     # 快速验证流程是否跑通
    python code/train.py --resume       # 从 last.pt 断点续训

任务书 11.3 节要求：超参集中在顶部 CONFIG 字典，且必须有
`if __name__ == "__main__":` 保护块——Windows 用 spawn 启动子进程，
没有保护块会无限递归重启。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许 `python code/train.py` 直接运行：把 code/ 加进模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DATASET_ROOT,
    LAST_WEIGHTS,
    RUN_NAME,
    RUNS_DIR,
    count_images,
    list_classes,
    model_fingerprint,
    rel,
    setup_env,
)

# 环境变量必须在导入 ultralytics 之前设置
setup_env()

try:
    from ultralytics import YOLO  # noqa: E402
    import torch  # noqa: E402
except ImportError as _e:  # 未装依赖时给出可执行的下一步，而不是一段 traceback
    print(
        f"[错误] 缺少依赖：{_e.name}\n"
        "请先激活虚拟环境并安装依赖：\n"
        "  .venv\\Scripts\\activate\n"
        "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
        "  pip install -r code/requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from None


# ============ 本组配置：按实际情况修改 ============
# 任务书 3.3 节要求组间模型至少三项不同。本组取值：
#   网络规模 yolo26n-cls / 输入 224 / 全量微调 / 默认 randaugment
# 若与别组撞车，优先改 model 与 imgsz，这两项对指纹影响最直接。
CONFIG = {
    "data_root": str(DATASET_ROOT),   # 数据集根目录，含 train/val/test
    "model": "yolo26n-cls.pt",        # 型号：n / s / m，按差异化要求选择
    "epochs": 50,                     # 训练轮数
    "imgsz": 224,                     # 输入尺寸：224 / 320 / 448
    "batch": 32,                      # 批大小，显存不足时减半
    "device": 0,                      # 0 表示第 1 块显卡；无显卡填 "cpu"
    "workers": 2,                     # 数据加载线程；Windows 建议 0—4
    "patience": 15,                   # 验证指标连续 N 轮不提升则早停
    "optimizer": "auto",              # auto / SGD / AdamW
    "lr0": 0.001,                     # 初始学习率（optimizer=auto 时被忽略）
    "dropout": 0.0,                   # 分类任务专用正则化，过拟合时可设 0.2
    "freeze": None,                   # 冻结前 N 层；None 表示全量微调
    "seed": 42,                       # 随机种子，保证可复现
    "pretrained": True,               # 使用 ImageNet 预训练权重
    "project": str(RUNS_DIR),         # 输出根目录
    "name": RUN_NAME,                 # 本次运行名，输出落在 project/name/
    "exist_ok": True,                 # 允许覆盖同名目录
    "plots": True,                    # 生成训练曲线与混淆矩阵图
    "cache": False,                   # 数据集小可设 True 加速
}

# 参与模型指纹的六项配置（任务书 3.3 节表格）
FINGERPRINT_KEYS = ("model", "imgsz", "freeze", "augment", "optimizer", "epochs")
FINGERPRINT_EXTRA = {"augment": "randaugment"}  # 增强策略走默认值，显式登记


def check_device(cfg: dict) -> None:
    """训练前确认设备可用，避免跑到一半才报错。"""
    if str(cfg["device"]).lower() == "cpu":
        print("[设备] 使用 CPU 训练，速度较慢但可正常完成")
        return
    if not torch.cuda.is_available():
        print("[警告] 未检测到可用 CUDA，自动切换为 CPU")
        cfg["device"] = "cpu"
        return
    props = torch.cuda.get_device_properties(cfg["device"])
    print(
        f"[设备] {torch.cuda.get_device_name(cfg['device'])}"
        f"，显存 {props.total_memory / 1024**3:.1f} GB"
    )


def check_dataset(cfg: dict) -> None:
    """训练前确认目录结构：train/val 齐全，类别一致，且确实有图片。"""
    data_root = Path(cfg["data_root"])
    if not data_root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在：{data_root}")

    class_map: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for sub in ("train", "val"):
        p = data_root / sub
        if not p.is_dir():
            raise FileNotFoundError(f"缺少 {sub} 目录：{p}")
        classes = list_classes(sub)
        if not classes:
            raise RuntimeError(f"{sub}/ 下没有任何类别文件夹")
        class_map[sub] = classes
        counts[sub] = sum(count_images(p / c) for c in classes)
        print(
            f"[检查] {sub}/ 下 {len(classes)} 个类别，共 {counts[sub]} 张图片"
        )

    if counts["train"] == 0:
        raise RuntimeError(
            "train/ 下没有任何图片。当前只是空目录框架，"
            "请先把采集的照片按类别放进 train/<类别>/ 再训练。"
        )
    if counts["val"] == 0:
        raise RuntimeError(
            "val/ 下没有任何图片，训练无法选择最优轮次。"
            "请按 8:1:1 划分验证集。"
        )

    if class_map["train"] != class_map["val"]:
        only_train = set(class_map["train"]) - set(class_map["val"])
        only_val = set(class_map["val"]) - set(class_map["train"])
        raise RuntimeError(
            "train 与 val 的类别文件夹不一致，会导致类别索引错位。\n"
            f"  仅 train 有：{sorted(only_train)}\n"
            f"  仅 val 有：{sorted(only_val)}"
        )

    test_dir = data_root / "test"
    if test_dir.is_dir():
        missing = set(class_map["train"]) - set(list_classes("test"))
        if missing:
            print(f"[警告] test/ 缺少类别文件夹：{sorted(missing)}")


def print_fingerprint(cfg: dict) -> str:
    """打印并返回模型指纹，供 README 第五章登记。"""
    fp_config = {k: cfg.get(k, FINGERPRINT_EXTRA.get(k)) for k in FINGERPRINT_KEYS}
    digest, payload = model_fingerprint(fp_config)
    print(f"\n[指纹] {digest}")
    print(f"[指纹] 配置明细：{payload}")
    return digest


def build_train_kwargs(cfg: dict) -> dict:
    """把 CONFIG 翻译成 model.train() 的关键字参数。"""
    return {
        "data": str(cfg["data_root"]),
        "epochs": cfg["epochs"],
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "device": cfg["device"],
        "workers": cfg["workers"],
        "patience": cfg["patience"],
        "optimizer": cfg["optimizer"],
        "lr0": cfg["lr0"],
        "dropout": cfg["dropout"],
        "freeze": cfg["freeze"],
        "seed": cfg["seed"],
        "pretrained": cfg["pretrained"],
        "project": str(cfg["project"]),
        "name": cfg["name"],
        "exist_ok": cfg["exist_ok"],
        "plots": cfg["plots"],
        "cache": cfg["cache"],
        "val": True,
        "save": True,
        "verbose": True,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="训练开封菊花分类模型")
    parser.add_argument("--epochs", type=int, help="覆盖 CONFIG 中的训练轮数")
    parser.add_argument("--imgsz", type=int, help="覆盖 CONFIG 中的输入尺寸")
    parser.add_argument("--batch", type=int, help="覆盖 CONFIG 中的批大小")
    parser.add_argument("--device", help='覆盖设备，如 "cpu" 或 0')
    parser.add_argument("--model", help="覆盖模型型号，如 yolo26s-cls.pt")
    parser.add_argument("--name", help="覆盖运行名，输出落在 results/runs/<name>/")
    parser.add_argument(
        "--resume", action="store_true", help="从 last.pt 断点续训"
    )
    args = parser.parse_args(argv)

    cfg = dict(CONFIG)
    for key in ("epochs", "imgsz", "batch", "device", "model", "name"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    # device 从命令行进来是字符串，纯数字要转回 int
    if isinstance(cfg["device"], str) and cfg["device"].isdigit():
        cfg["device"] = int(cfg["device"])

    if args.resume:
        if not LAST_WEIGHTS.is_file():
            raise FileNotFoundError(
                f"找不到 {rel(LAST_WEIGHTS)}，至少完整跑完 1 个 epoch 才有 last.pt 可续训"
            )
        print(f"[续训] 从 {rel(LAST_WEIGHTS)} 恢复")
        model = YOLO(str(LAST_WEIGHTS))
        model.train(resume=True)
        print("\n=== 续训完成 ===")
        print(f"最优权重：{model.trainer.best}")
        return

    check_device(cfg)
    check_dataset(cfg)
    print_fingerprint(cfg)

    print(f"\n[模型] 加载 {cfg['model']}")
    model = YOLO(cfg["model"])

    print(f"[训练] 输出目录 {rel(Path(cfg['project']) / cfg['name'])}")
    model.train(**build_train_kwargs(cfg))

    print("\n=== 训练完成 ===")
    print(f"最优权重：{model.trainer.best}")
    print(f"最后一轮：{model.trainer.last}")
    print(f"训练曲线：{rel(Path(cfg['project']) / cfg['name'] / 'results.png')}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as e:
        # 数据/路径类问题给出干净提示，不甩一屏 traceback
        print(f"\n[错误] {e}", file=sys.stderr)
        raise SystemExit(1)
