"""数据准备工具：批量重命名、去重、按植株划分到 train/val/test。

这是任务书加分项"提供可复用的工具脚本"对应的实现。三个子命令可以单独用：

    # 1. 按 {类别}_{三位序号}.jpg 批量重命名 _raw/<类别>/ 下的原图
    python code/prepare_dataset.py rename --src _raw/01_huangse_pingban

    # 2. 按文件哈希去重（完全相同的图片只留一张）
    python code/prepare_dataset.py dedupe --src _raw/01_huangse_pingban

    # 3. 按植株分组划分到 train/val/test（防泄漏的关键一步）
    python code/prepare_dataset.py split --src _raw/01_huangse_pingban \
        --category 01_huangse_pingban --per-plant 9

划分原则（任务书 7.2 节）：同一次拍摄、同一株花的多张照片必须整体进入
同一个集合。本脚本以"植株"为单位分配，绝不按图片随机打乱——随机打乱会让
同一株的正面照进训练集、侧面照进测试集，测试准确率虚高十几个百分点。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATASET_ROOT, iter_images, rel  # noqa: E402

# 任务书 7.1 节推荐 8:1:1
DEFAULT_RATIO = (8, 1, 1)


def cmd_rename(args: argparse.Namespace) -> int:
    """把目录下的图片按 {类别}_{三位序号}.jpg 重命名。

    序号从 001 连续编到该类最后一张，中间不跳号。重名会导致拷贝时覆盖，
    是最隐蔽的数据丢失方式，所以先检查目标名是否已存在。
    """
    src = Path(args.src)
    if not src.is_dir():
        print(f"[错误] 目录不存在：{src}", file=sys.stderr)
        return 1

    category = args.category or src.name
    images = iter_images(src)
    if not images:
        print(f"[提示] {rel(src)} 下没有图片")
        return 0

    # 先算出全部目标名，确认没有冲突再动手，避免改到一半失败
    plan: list[tuple[Path, Path]] = []
    for i, img in enumerate(images, start=1):
        target = src / f"{category}_{i:03d}{img.suffix.lower()}"
        if target.exists() and target != img:
            print(f"[错误] 目标文件已存在，可能重复执行过：{target.name}", file=sys.stderr)
            return 1
        plan.append((img, target))

    if args.dry_run:
        for old, new in plan[:10]:
            print(f"  {old.name} → {new.name}")
        if len(plan) > 10:
            print(f"  ...（共 {len(plan)} 个）")
        print("[试运行] 未实际改名，去掉 --dry-run 执行")
        return 0

    # 两段式改名：先改成临时名，再改成目标名，避免 a→b、b→c 的连环覆盖
    for old, _ in plan:
        old.rename(old.with_name(old.name + ".tmp"))
    for old, new in plan:
        old.with_name(old.name + ".tmp").rename(new)

    print(f"已重命名 {len(plan)} 个文件，类别 {category}")
    return 0


def cmd_dedupe(args: argparse.Namespace) -> int:
    """按文件哈希去重，完全相同的图片只保留第一张。"""
    src = Path(args.src)
    if not src.is_dir():
        print(f"[错误] 目录不存在：{src}", file=sys.stderr)
        return 1

    seen: dict[str, Path] = {}
    dupes: list[Path] = []
    for img in iter_images(src):
        digest = hashlib.md5(img.read_bytes()).hexdigest()
        if digest in seen:
            dupes.append(img)
        else:
            seen[digest] = img

    if not dupes:
        print(f"{rel(src)}：{len(seen)} 张图片，无重复")
        return 0

    for d in dupes:
        print(f"  重复：{d.name}（与 {seen[hashlib.md5(d.read_bytes()).hexdigest()].name} 相同）")
        if not args.dry_run:
            d.unlink()

    action = "已删除" if not args.dry_run else "可删除（试运行）"
    print(f"{action} {len(dupes)} 张重复图片，保留 {len(seen)} 张")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    """按植株分组把图片划分到 train/val/test。

    以植株为单位分配集合，而不是以图片为单位。若没有植株编号，按拍摄
    顺序分批——拍摄时"一株拍完再换下一株"，连续序号天然对应植株。

    算法与任务书 7.2 节的例子一致：整株进 train，剩下的尾株在 val 与 test
    之间按比例切开。例如 5 株 × 9 张 = 45 张：P01—P04 进 train（36 张），
    P05 的前 4 张进 val、后 5 张进 test（val 4 / test 5）。
    这样既守住了"同株不同集合"的红线（只有 val↔test 边界会切开一株，
    不影响训练），又保证测试集不会为空。
    """
    src = Path(args.src)
    if not src.is_dir():
        print(f"[错误] 目录不存在：{src}", file=sys.stderr)
        return 1

    dest_root = Path(args.dst) if args.dst else DATASET_ROOT
    category = args.category or src.name
    images = iter_images(src)
    if not images:
        print(f"[提示] {rel(src)} 下没有图片")
        return 0

    per_plant = args.per_plant
    if per_plant < 1:
        print("[错误] --per-plant 必须 ≥ 1", file=sys.stderr)
        return 1

    # 按拍摄顺序切分成植株组
    plants: list[list[Path]] = [
        images[i : i + per_plant] for i in range(0, len(images), per_plant)
    ]
    total = len(images)
    total_plants = len(plants)
    r_train, r_val, r_test = args.ratio
    ratio_sum = r_train + r_val + r_test

    # 第一步：整株进 train，直到达到训练目标张数；但要给 val/test 至少留一株
    train_target = total * r_train / ratio_sum
    train_imgs: list[Path] = []
    consumed = 0
    idx = 0
    while idx < total_plants:
        plants_left_after = total_plants - (idx + 1)
        if plants_left_after < 1:
            break
        if consumed + len(plants[idx]) > train_target:
            break
        train_imgs.extend(plants[idx])
        consumed += len(plants[idx])
        idx += 1

    # 第二步：剩余图片按 val:test 比例切开，测试集优先满足 ≥ 5 张
    remaining = [p for plant in plants[idx:] for p in plant]
    if not remaining:
        print(
            f"[警告] 共 {total_plants} 株，全部进了 train，测试集为空。"
            "任务书要求每类测试集 ≥ 5 张，请增加拍摄株数。"
        )
        val_imgs: list[Path] = []
        test_imgs: list[Path] = []
    else:
        min_test = 5 if len(remaining) >= 5 else max(1, len(remaining) // 2)
        val_target = len(remaining) * r_val / (r_val + r_test)
        val_count = int(round(val_target))
        # 夹在 [1, len(remaining) - min_test] 之间
        upper = max(0, len(remaining) - min_test)
        val_count = max(1, min(val_count, upper)) if upper >= 1 else 0
        val_imgs = remaining[:val_count]
        test_imgs = remaining[val_count:]

    counts = {"train": 0, "val": 0, "test": 0}
    assignment = [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]
    for split, split_images in assignment:
        dest_dir = dest_root / split / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        for img in split_images:
            dest = dest_dir / img.name
            if dest.exists():
                print(f"[跳过] 已存在：{rel(dest)}")
                continue
            if args.dry_run:
                print(f"  [试运行] {img.name} → {split}/{category}/")
            else:
                # 复制而不是移动：保留 _raw/ 原目录作为备份，直到提交前再删
                shutil.copy2(img, dest)
            counts[split] += 1

    print(f"\n类别 {category}：{total_plants} 株，{total} 张")
    print(
        f"  train {counts['train']} 张 / val {counts['val']} 张 / test {counts['test']} 张"
    )
    if counts["test"] < 5:
        print(f"  [警告] 测试集只有 {counts['test']} 张，少于要求的 5 张")
    if counts["train"] < 1:
        print("  [警告] 训练集为空，请减少 --per-plant 或增加拍摄量")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="数据准备工具：重命名 / 去重 / 按植株划分"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_rename = sub.add_parser("rename", help="批量重命名为 {类别}_{三位序号}.jpg")
    p_rename.add_argument("--src", required=True, help="待重命名的目录")
    p_rename.add_argument("--category", help="类别文件夹名，默认取目录名")
    p_rename.add_argument("--dry-run", action="store_true", help="只打印不执行")
    p_rename.set_defaults(func=cmd_rename)

    p_dedupe = sub.add_parser("dedupe", help="按文件哈希去重")
    p_dedupe.add_argument("--src", required=True, help="待去重的目录")
    p_dedupe.add_argument("--dry-run", action="store_true", help="只打印不执行")
    p_dedupe.set_defaults(func=cmd_dedupe)

    p_split = sub.add_parser("split", help="按植株分组划分到 train/val/test")
    p_split.add_argument("--src", required=True, help="原图目录")
    p_split.add_argument(
        "--dst", help="数据集根目录，默认项目根目录（含 train/val/test）"
    )
    p_split.add_argument("--category", help="类别文件夹名，默认取目录名")
    p_split.add_argument(
        "--per-plant", type=int, default=9, help="每株拍摄张数（默认 9）"
    )
    p_split.add_argument(
        "--ratio",
        type=int,
        nargs=3,
        default=list(DEFAULT_RATIO),
        metavar=("TRAIN", "VAL", "TEST"),
        help="划分比例，默认 8 1 1",
    )
    p_split.add_argument("--dry-run", action="store_true", help="只打印不执行")
    p_split.set_defaults(func=cmd_split)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
