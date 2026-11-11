"""组间数据集相似度核查（dHash 感知哈希）。

用法：
    python code/compare_datasets.py <本组数据集> <他组数据集> [阈值]
    python code/compare_datasets.py D:\\groupA D:\\groupB 8

任务书附录 C：相似度 = 重复图片数 / min(A 张数, B 张数) × 100%。
判定要双指标——总相似度看整体重合，完全重复率看是否有整批复制：
    总相似度   < 30% 通过 / 30%—70% 警告 / ≥ 70% 不通过
    完全重复率 < 10% 通过 / 10%—20% 警告 / ≥ 20% 不通过
任一指标落入"不通过"即判不通过。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 独立脚本，不依赖项目内模块：自己把输出统一成 UTF-8，
# 否则 Windows 默认 GBK 控制台下中文显示为乱码。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

try:
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover
    print("缺少依赖，请先执行：pip install pillow numpy", file=sys.stderr)
    raise SystemExit(1)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def dhash(path: Path, size: int = 8) -> int:
    """差值哈希：转灰度 → 缩到 (size+1, size) → 相邻像素比较 → 64 位整数。

    对压缩、缩放、轻微调色不敏感，但对内容变化敏感——两朵不同的花
    会得到差异很大的哈希。这正是不能用文件 MD5 的原因。
    """
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = np.asarray(im, dtype=np.int16)
    diff = px[:, 1:] > px[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits


def collect(root: Path) -> list[tuple[Path, int]]:
    """遍历目录下所有图片，返回 [(路径, 哈希)]。"""
    items: list[tuple[Path, int]] = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            items.append((p, dhash(p)))
        except (OSError, ValueError) as e:
            print(f"[跳过] {p.name}：{e}")
    return items


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def similarity_detail(
    root_a: Path, root_b: Path, loose: int = 8, strict: int = 2
) -> dict | None:
    """双指标相似度：loose 判总重合，strict 判近乎逐像素复制。"""
    a = collect(root_a)
    b = collect(root_b)
    if not a or not b:
        print("某一侧没有找到图片")
        return None

    dup = dup_strict = 0
    for _, ha in a:
        best = min(hamming(ha, hb) for _, hb in b)
        if best <= loose:
            dup += 1
        if best <= strict:
            dup_strict += 1

    base = min(len(a), len(b))
    return {
        "count_a": len(a),
        "count_b": len(b),
        "base": base,
        "dup": dup,
        "dup_strict": dup_strict,
        "loose_pct": round(dup / base * 100, 1),
        "strict_pct": round(dup_strict / base * 100, 1),
    }


def verdict(pct: float, pass_line: float, warn_line: float) -> str:
    if pct < pass_line:
        return "通过"
    if pct < warn_line:
        return "警告"
    return "不通过"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="核查两组数据集的相似度")
    parser.add_argument("root_a", help="本组数据集目录")
    parser.add_argument("root_b", help="他组数据集目录")
    parser.add_argument("--loose", type=int, default=8, help="总相似度阈值（默认 8）")
    parser.add_argument(
        "--strict", type=int, default=2, help="完全重复阈值（默认 2）"
    )
    args = parser.parse_args(argv)

    res = similarity_detail(Path(args.root_a), Path(args.root_b), args.loose, args.strict)
    if res is None:
        return 1

    v_loose = verdict(res["loose_pct"], 30, 70)
    v_strict = verdict(res["strict_pct"], 10, 20)

    print("=" * 52)
    print(f"阈值（汉明距离）  : 总相似度 ≤ {args.loose}，完全重复 ≤ {args.strict}")
    print(f"A 组张数          : {res['count_a']}")
    print(f"B 组张数          : {res['count_b']}")
    print(f"基准张数          : {res['base']}")
    print("-" * 52)
    print(f"重复图片数        : {res['dup']}")
    print(f"总相似度          : {res['loose_pct']}%  → {v_loose}")
    print(f"完全重复数        : {res['dup_strict']}")
    print(f"完全重复率        : {res['strict_pct']}%  → {v_strict}")
    print("-" * 52)
    if "不通过" in (v_loose, v_strict):
        print("判定：不通过（数据集雷同）")
        print("=" * 52)
        return 2
    if "警告" in (v_loose, v_strict):
        print("判定：警告（需在 README 说明重合原因）")
        print("=" * 52)
        return 0
    print("判定：通过（正常重合）")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
