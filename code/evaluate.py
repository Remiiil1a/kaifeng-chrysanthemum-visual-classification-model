"""三集合评估脚本：train / val / test 分别测一次 top1、top5。

用法：
    python code/evaluate.py
    python code/evaluate.py --weights results/runs/chrys_yolo26n/weights/best.pt

任务书 11.6 节：best.pt 是在验证集上挑出来的，最终成绩取测试集，且只取一次。
结果写入 results/eval_summary.json，同时打印模型指纹供组间核查。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    BEST_WEIGHTS,
    DATASET_ROOT,
    RESULTS_DIR,
    SPLITS,
    model_fingerprint,
    rel,
    setup_env,
)

setup_env()

try:
    from ultralytics import YOLO  # noqa: E402
except ImportError as _e:
    print(
        f"[错误] 缺少依赖：{_e.name}，请先激活虚拟环境并安装 code/requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from None

# 本组的模型配置，必须与 train.py 的 CONFIG 保持一致，
# 否则算出来的指纹对不上，组间核查会误判。
FINGERPRINT = {
    "model": "yolo26n-cls",
    "imgsz": 224,
    "freeze": None,
    "augment": "randaugment",
    "optimizer": "auto",
    "epochs": 50,
}


def evaluate(weights: Path, data_root: Path, imgsz: int, batch: int) -> dict:
    model = YOLO(str(weights))

    print("=== 类别映射 ===")
    for idx, name in model.names.items():
        print(f"  {idx}: {name}")

    print("\n=== 三个集合的准确率 ===")
    summary: dict[str, dict] = {}
    for split in SPLITS:
        split_dir = data_root / split
        if not split_dir.is_dir():
            print(f"  {split:>5}: 跳过（目录不存在）")
            continue
        metrics = model.val(
            data=str(data_root), split=split, imgsz=imgsz, batch=batch, verbose=False
        )
        summary[split] = {
            "top1": round(float(metrics.top1), 4),
            "top5": round(float(metrics.top5), 4),
        }
        print(f"  {split:>5}: top1 = {metrics.top1:.4f}   top5 = {metrics.top5:.4f}")

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="在三个集合上评估分类模型")
    parser.add_argument("--weights", default=str(BEST_WEIGHTS), help="权重路径")
    parser.add_argument("--data", default=str(DATASET_ROOT), help="数据集根目录")
    parser.add_argument("--imgsz", type=int, default=FINGERPRINT["imgsz"])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--out", default=str(RESULTS_DIR / "eval_summary.json"), help="结果输出路径"
    )
    args = parser.parse_args(argv)

    weights = Path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(
            f"找不到权重 {weights}，请先运行 code/train.py"
        )

    summary = evaluate(weights, Path(args.data), args.imgsz, args.batch)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果已保存：{rel(out)}")

    digest, payload = model_fingerprint(FINGERPRINT)
    print(f"模型指纹：{digest}")
    print(f"  配置明细：{payload}")

    # 任务书 11.6 节：train 与 test 差距超过 15 个百分点要在报告里分析
    if "train" in summary and "test" in summary:
        gap = (summary["train"]["top1"] - summary["test"]["top1"]) * 100
        if gap > 15:
            print(
                f"\n[提示] train 与 test 的 top1 相差 {gap:.1f} 个百分点，"
                "超过 15 个百分点，需在 README 第五部分如实分析过拟合原因"
            )


if __name__ == "__main__":
    main()
