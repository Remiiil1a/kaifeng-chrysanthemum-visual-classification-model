"""从图片 EXIF 批量生成 labels.csv。

用法：
    python code/build_labels.py                    # 扫描 train/val/test
    python code/build_labels.py --root . --out labels.csv
    python code/build_labels.py --plant-rule seq   # 按序号区间推断 plant_id

任务书 8.3 节：手工填 450 行既耗时又必然出错，且无法保证经纬度真实。
本脚本读取每张图的 EXIF，缺 GPS 的图片会逐条打印出来——这是提交前
必须清零的清单（任务书要求 GPS 完整率 100%）。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import LABEL_FIELDS, LABELS_CSV, ROOT, SPLITS, iter_images, rel  # noqa: E402

try:
    from PIL import Image
    from PIL.ExifTags import GPSTAGS
except ImportError:  # pragma: no cover - 环境没装 Pillow 时给出可执行的提示
    print("缺少 Pillow，请先执行：pip install pillow", file=sys.stderr)
    raise SystemExit(1)


def dms_to_deg(dms, ref: str) -> float:
    """度分秒转十进制度，南纬西经取负。"""
    d, m, s = (float(x) for x in dms)
    deg = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        deg = -deg
    return round(deg, 6)


def read_exif(path: Path) -> dict | None:
    """读取单张图片的经纬度、海拔与拍摄时间；读不到 GPS 返回 None。"""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return None
            gps_info = exif.get_ifd(0x8825)
            if not gps_info:
                return None
            gps = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}
            try:
                lat = dms_to_deg(gps["GPSLatitude"], gps["GPSLatitudeRef"])
                lon = dms_to_deg(gps["GPSLongitude"], gps["GPSLongitudeRef"])
            except (KeyError, ValueError, TypeError):
                return None

            # 306 = DateTime，格式 "YYYY:MM:DD HH:MM:SS"，只替换前两个冒号
            dt = str(exif.get(306, "")).replace(":", "-", 2)
            alt = gps.get("GPSAltitude")
            device = str(exif.get(272, "")).strip()  # 272 = Model
            return {
                "latitude": lat,
                "longitude": lon,
                "altitude": round(float(alt), 1) if alt else "",
                "datetime": dt,
                "device": device,
            }
    except (OSError, ValueError):
        return None


def infer_plant_id(filename: str, rule: str, per_plant: int) -> str:
    """推断植株编号。

    任务书 7.2 节：划分必须以植株为单位，否则同一株的正面照进训练集、
    侧面照进测试集，测试准确率会虚高十几个百分点。

    拍摄时若按"一株拍完再换下一株"的习惯，连续序号天然对应植株，
    因此 seq 规则按每 per_plant 张归为一株。
    """
    if rule == "none":
        return ""
    stem = Path(filename).stem
    try:
        seq = int(stem.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return ""
    plant_no = (seq - 1) // per_plant + 1
    return f"P{plant_no:02d}"


def build_labels(
    root: Path, out_csv: Path, plant_rule: str, per_plant: int
) -> tuple[int, list[str]]:
    rows: list[dict] = []
    missing: list[str] = []

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            print(f"[跳过] 缺少目录 {split}/")
            continue
        for cat_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            for img in iter_images(cat_dir):
                info = read_exif(img)
                if info is None:
                    missing.append(f"{split}/{cat_dir.name}/{img.name}")
                    continue
                rows.append(
                    {
                        "filename": img.name,
                        "split": split,
                        "category": cat_dir.name,
                        "plant_id": infer_plant_id(img.name, plant_rule, per_plant),
                        "latitude": info["latitude"],
                        "longitude": info["longitude"],
                        "altitude": info["altitude"],
                        "datetime": info["datetime"],
                        "device": info["device"],
                    }
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig：Excel 直接双击打开不乱码
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), missing


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="从 EXIF 生成 labels.csv")
    parser.add_argument("--root", default=str(ROOT), help="数据集根目录")
    parser.add_argument("--out", default=str(LABELS_CSV), help="输出 CSV 路径")
    parser.add_argument(
        "--plant-rule",
        choices=("seq", "none"),
        default="seq",
        help="plant_id 推断规则：seq 按序号区间，none 留空待手工补填",
    )
    parser.add_argument(
        "--per-plant", type=int, default=9, help="每株拍摄张数，用于 seq 规则"
    )
    args = parser.parse_args(argv)

    out_csv = Path(args.out)
    count, missing = build_labels(Path(args.root), out_csv, args.plant_rule, args.per_plant)

    print(f"已写出 {rel(out_csv)}，共 {count} 条记录")
    if missing:
        print(f"\n[缺少 GPS] 共 {len(missing)} 张，这些图片不会进入 labels.csv：")
        for item in missing:
            print(f"  {item}")
        print(
            "\n处理办法：确认是否用了微信/QQ 传图（会剥离 EXIF），"
            "或拍摄时定位未锁定。任务书要求 GPS 完整率 100%。"
        )
    else:
        print("全部图片均含 GPS 信息")


if __name__ == "__main__":
    main()
