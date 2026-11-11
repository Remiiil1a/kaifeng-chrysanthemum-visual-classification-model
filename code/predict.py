"""单张 / 批量预测脚本，用于演示与答辩。

用法：
    python code/predict.py test/01_huangse_pingban
    python code/predict.py test/01_huangse_pingban/01_huangse_pingban_039.jpg
    python code/predict.py test --save          # 保存带标注的图片

任务书 11.7 节：类别名一律用 model.names 反查，绝不硬编码索引。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BEST_WEIGHTS, IMAGE_EXTS, rel, setup_env  # noqa: E402

setup_env()

try:
    from ultralytics import YOLO  # noqa: E402
except ImportError as _e:
    print(
        f"[错误] 缺少依赖：{_e.name}，请先激活虚拟环境并安装 code/requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from None


def collect_targets(target: Path) -> list[Path]:
    """把命令行参数展开成待预测的图片列表。"""
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(
            p for p in target.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
    raise FileNotFoundError(f"路径不存在：{target}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="预测菊花类别")
    parser.add_argument("target", help="图片文件或目录")
    parser.add_argument("--weights", default=str(BEST_WEIGHTS), help="权重路径")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--topk", type=int, default=5, help="每张图打印前 k 个候选")
    parser.add_argument("--save", action="store_true", help="保存带预测标注的图片")
    args = parser.parse_args(argv)

    weights = Path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"找不到权重 {weights}，请先运行 code/train.py")

    targets = collect_targets(Path(args.target))
    if not targets:
        raise RuntimeError(f"{args.target} 下没有可预测的图片")

    model = YOLO(str(weights))
    print(f"=== 类别映射（共 {len(model.names)} 类）===")
    for idx, name in model.names.items():
        print(f"  {idx}: {name}")

    print(f"\n=== 预测 {len(targets)} 张 ===")
    results = model(
        [str(p) for p in targets],
        imgsz=args.imgsz,
        save=args.save,
        verbose=False,
    )

    correct = 0
    for r in results:
        idx = int(r.probs.top1)
        conf = float(r.probs.top1conf)
        name = model.names[idx]
        path = Path(r.path)
        # 目录结构本身就是标注：父文件夹名即真实类别
        truth = path.parent.name
        hit = truth == name
        correct += hit
        mark = "OK " if hit else "ERR"
        print(f"[{mark}] {path.name:<40} → {name:<26} {conf:.4f}")

        if args.topk > 1:
            top = r.probs.top5[: args.topk]
            detail = "  ".join(
                f"{model.names[int(i)]}({float(c):.3f})" for i, c in zip(top, r.probs.top5conf[: args.topk])
            )
            print(f"        top{args.topk}: {detail}")

    if all(p.parent.name in model.names.values() for p in targets):
        acc = correct / len(targets)
        print(f"\n按父文件夹名核对：{correct}/{len(targets)} 正确，准确率 {acc:.4f}")
    if args.save:
        print(f"\n带标注的图片已保存到 {rel(Path('runs/classify/predict'))}")


if __name__ == "__main__":
    main()
