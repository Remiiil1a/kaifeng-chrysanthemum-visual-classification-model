"""数据集结构自查，训练前必跑一次。

用法：
    python code/check_dataset.py
    python code/check_dataset.py --strict     # 有任一问题就以非零码退出

覆盖任务书 7.3 节与第十三章自检清单里能自动化的部分：
  1. train/val/test 三个目录是否齐全
  2. 三个子集的类别文件夹是否齐全且名称完全一致
  3. 每个类别在各子集的张数，测试集是否 ≥ 5 张
  4. 文件名是否全局唯一、是否符合 {类别}_{三位序号}.jpg 规范
  5. 图片是否直接位于类别文件夹下（无中间层）
  6. labels.csv 行数是否等于图片总数
  7. 同一 plant_id 是否跨越多个集合（数据泄漏）
  8. 经纬度是否落在开封范围（约 34.7—34.9°N，114.2—114.4°E）
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DATASET_ROOT,
    SPLITS,
    count_images,
    iter_images,
    list_classes,
    rel,
)

# 任务书 4.3 节：{类别文件夹名}_三位序号.jpg
NAME_PATTERN = re.compile(r"^(?P<cat>[a-z0-9_]+)_(?P<seq>\d{3})\.(jpg|jpeg|png)$", re.I)

# 开封市区大致范围，任务书第十三章第 8 项
KAIFENG_LAT = (34.7, 34.9)
KAIFENG_LON = (114.2, 114.4)

MIN_TEST_PER_CLASS = 5
MIN_CLASSES = 10


class Report:
    """收集检查结果，最后统一打印，避免边查边刷屏。"""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> int:
        print("\n" + "=" * 64)
        if self.errors:
            print(f"[不通过] {len(self.errors)} 项错误：")
            for i, msg in enumerate(self.errors, 1):
                print(f"  {i}. {msg}")
        if self.warnings:
            print(f"[警告] {len(self.warnings)} 项：")
            for i, msg in enumerate(self.warnings, 1):
                print(f"  {i}. {msg}")
        if not self.errors and not self.warnings:
            print("[通过] 未发现问题")
        elif not self.errors:
            print("[通过] 无阻断性错误，警告项建议处理")
        print("=" * 64)
        return 1 if self.errors else 0


def check_structure(root: Path, report: Report) -> dict[str, list[str]]:
    """检查三个子集目录与类别文件夹的一致性。"""
    class_map: dict[str, list[str]] = {}
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            report.error(f"缺少目录 {split}/")
            class_map[split] = []
            continue
        classes = list_classes(split)
        class_map[split] = classes
        if not classes:
            report.error(f"{split}/ 下没有任何类别文件夹")

    present = [s for s in SPLITS if class_map.get(s)]
    if len(present) >= 2:
        base_split = present[0]
        base = set(class_map[base_split])
        for split in present[1:]:
            other = set(class_map[split])
            if other != base:
                only_base = sorted(base - other)
                only_other = sorted(other - base)
                report.error(
                    f"{split}/ 与 {base_split}/ 的类别文件夹不一致"
                    f"（仅 {base_split} 有：{only_base}；仅 {split} 有：{only_other}）"
                )

    if class_map.get("train") and len(class_map["train"]) < MIN_CLASSES:
        report.error(
            f"类别数 {len(class_map['train'])} 少于任务书要求的 {MIN_CLASSES} 类"
        )

    return class_map


def check_counts(root: Path, class_map: dict[str, list[str]], report: Report) -> dict:
    """统计各类别张数，检查测试集下限与空类别。"""
    table: dict[str, dict[str, int]] = defaultdict(dict)
    for split in SPLITS:
        for cat in class_map.get(split, []):
            table[cat][split] = count_images(root / split / cat)

    cats = sorted(table)
    print(f"\n类别数：{len(cats)}")
    print(f"{'类别':<28}{'train':>7}{'val':>7}{'test':>7}{'合计':>7}")
    print("-" * 64)
    totals = {"train": 0, "val": 0, "test": 0}
    for cat in cats:
        row = table[cat]
        t, v, s = row.get("train", 0), row.get("val", 0), row.get("test", 0)
        totals["train"] += t
        totals["val"] += v
        totals["test"] += s
        flag = ""
        if s < MIN_TEST_PER_CLASS:
            flag = f"  ← 测试集不足 {MIN_TEST_PER_CLASS} 张"
            report.error(f"{cat} 的测试集只有 {s} 张，少于 {MIN_TEST_PER_CLASS} 张")
        if t == 0:
            report.warn(f"{cat} 的训练集为空，该类别无法学习")
        print(f"{cat:<28}{t:>7}{v:>7}{s:>7}{t + v + s:>7}{flag}")
    print("-" * 64)
    print(
        f"{'合计':<28}{totals['train']:>7}{totals['val']:>7}{totals['test']:>7}"
        f"{sum(totals.values()):>7}"
    )

    total = sum(totals.values())
    if total == 0:
        report.warn("数据集里还没有任何图片，当前只是空框架")
    elif total < 300:
        report.warn(f"总张数 {total} 少于任务书要求的 300 张")
    return table


def check_filenames(root: Path, class_map: dict[str, list[str]], report: Report) -> None:
    """检查文件名规范、全局唯一性与目录层级。"""
    seen: dict[str, str] = {}
    bad_pattern: list[str] = []
    nested: list[str] = []

    for split in SPLITS:
        for cat in class_map.get(split, []):
            cat_dir = root / split / cat
            for img in iter_images(cat_dir):
                key = img.name.lower()
                if key in seen:
                    report.error(
                        f"文件名重复：{img.name} 同时出现在 {seen[key]} 与 {split}/{cat}"
                    )
                else:
                    seen[key] = f"{split}/{cat}"

                m = NAME_PATTERN.match(img.name)
                if not m:
                    bad_pattern.append(f"{split}/{cat}/{img.name}")
                elif m.group("cat") != cat:
                    report.error(
                        f"文件名前缀与所在类别不符：{split}/{cat}/{img.name}"
                    )

            # 类别文件夹下不应再有子目录（任务书 6.1 节第 3 条）
            for sub in cat_dir.iterdir():
                if sub.is_dir():
                    nested.append(f"{split}/{cat}/{sub.name}/")

    if bad_pattern:
        report.warn(
            f"{len(bad_pattern)} 个文件名不符合 {{类别}}_{{三位序号}}.jpg 规范，"
            f"例如：{bad_pattern[:3]}"
        )
    if nested:
        report.error(
            f"类别文件夹下存在多余子目录，图片必须直接放在类别文件夹下：{nested[:5]}"
        )


def check_labels(root: Path, table: dict, report: Report) -> None:
    """检查 labels.csv 的行数、字段、泄漏与经纬度范围。"""
    csv_path = root / "labels.csv"
    if not csv_path.is_file():
        report.warn("还没有 labels.csv，运行 code/build_labels.py 生成")
        return

    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    total_images = sum(sum(row.values()) for row in table.values())
    if total_images and len(rows) != total_images:
        report.error(
            f"labels.csv 有 {len(rows)} 行，但图片总数是 {total_images} 张，两者必须相等"
        )

    # 同一植株跨越多个集合 = 数据泄漏。但严重程度不同：
    #   train↔val / train↔test：模型训练时见过这株花，测试结果虚高 → 红线
    #   val↔test：只是尾株被切开（任务书 7.2 节自己的例子：P05 前 4 张进
    #             val、后 5 张进 test），不影响训练，可接受但要记录
    plant_splits: dict[str, set[str]] = defaultdict(set)
    out_of_range: list[str] = []
    empty_plant = 0

    for row in rows:
        plant = (row.get("plant_id") or "").strip()
        if plant:
            plant_splits[plant].add(row.get("split", ""))
        else:
            empty_plant += 1

        try:
            lat = float(row.get("latitude") or "")
            lon = float(row.get("longitude") or "")
        except ValueError:
            out_of_range.append(f"{row.get('filename')}（经纬度无法解析）")
            continue
        if not (KAIFENG_LAT[0] <= lat <= KAIFENG_LAT[1]) or not (
            KAIFENG_LON[0] <= lon <= KAIFENG_LON[1]
        ):
            out_of_range.append(f"{row.get('filename')}（{lat}, {lon}）")

    hard_leak: dict[str, list[str]] = {}
    soft_cross: dict[str, list[str]] = {}
    for plant, splits in plant_splits.items():
        if len(splits) <= 1:
            continue
        if "train" in splits:
            hard_leak[plant] = sorted(splits)
        else:
            soft_cross[plant] = sorted(splits)

    if hard_leak:
        report.error(
            f"发现数据泄漏：{len(hard_leak)} 个植株同时出现在 train 与其他集合，"
            f"例如 {list(hard_leak.items())[:3]}"
        )
    if soft_cross:
        report.warn(
            f"{len(soft_cross)} 个植株跨了 val 与 test（不影响训练，属尾株切分），"
            f"例如 {list(soft_cross.items())[:3]}"
        )
    if empty_plant:
        report.warn(f"{empty_plant} 行的 plant_id 为空，无法核验是否泄漏")
    if out_of_range:
        report.error(
            f"{len(out_of_range)} 张图片的经纬度不在开封范围内"
            f"（{KAIFENG_LAT[0]}—{KAIFENG_LAT[1]}°N，"
            f"{KAIFENG_LON[0]}—{KAIFENG_LON[1]}°E），"
            f"例如：{out_of_range[:3]}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="数据集结构自查")
    parser.add_argument("--root", default=str(DATASET_ROOT), help="数据集根目录")
    parser.add_argument(
        "--strict", action="store_true", help="有错误时以非零退出码结束"
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    print("=" * 64)
    print(f"数据集根目录：{rel(root)}")

    report = Report()
    class_map = check_structure(root, report)
    table = check_counts(root, class_map, report)
    check_filenames(root, class_map, report)
    check_labels(root, table, report)

    code = report.summary()
    return code if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
