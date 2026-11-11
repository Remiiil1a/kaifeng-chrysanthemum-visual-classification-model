"""自测脚本：用合成图片跑通 rename → split → build_labels → check_dataset。

合成图片带真实 EXIF GPS，用于在没有真实采集数据时验证整条流水线。
只在临时目录里操作，不污染项目仓库。

用法：python code/selftest.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from common import IMAGE_EXTS  # noqa: E402

try:
    from PIL import Image
except ImportError:
    print("缺少 Pillow，无法自测：pip install pillow")
    raise SystemExit(1)

CATEGORIES = [
    "01_huangse_pingban",
    "02_baise_guanban",
    "03_lvse_jiban",
    "04_fense_lianzuo",
    "05_hongse_qiuxing",
    "06_fuse_feiwu",
    "07_huangse_guanban",
    "08_baise_pingban",
    "09_zise_chiban",
    "10_shuangse_tuogui",
]

IMAGES_PER_CATEGORY = 30
PER_PLANT = 9


def make_exif(lat: float, lon: float):
    """构造含 GPS 与拍摄时间的 EXIF 块。

    注意 Pillow 写 GPS 有理数必须用 IFDRational，直接给 (num, den) 元组
    会被当成"多值"而报错。
    """
    from PIL.TiffImagePlugin import IFDRational

    exif = Image.Exif()

    def dms(value: float):
        abs_v = abs(value)
        deg = int(abs_v)
        minute_f = (abs_v - deg) * 60
        minute = int(minute_f)
        sec = (minute_f - minute) * 60
        return (
            IFDRational(deg, 1),
            IFDRational(minute, 1),
            IFDRational(round(sec * 1000), 1000),
        )

    gps = {
        0: b"\x02\x02\x00\x00",  # GPSVersionID
        1: "N" if lat >= 0 else "S",
        2: dms(lat),
        3: "E" if lon >= 0 else "W",
        4: dms(lon),
        5: b"\x00",  # GPSAltitudeRef：0 = 海平面以上
        6: IFDRational(725, 10),  # GPSAltitude
    }
    exif[0x8825] = gps
    exif[36867] = "2026:11:08 09:14:22"  # DateTimeOriginal
    exif[306] = "2026:11:08 09:14:22"  # DateTime
    exif[272] = "Selftest Cam"  # Model
    return exif


def make_images(raw_dir: Path, category: str, count: int) -> None:
    out = raw_dir / category
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, count + 1):
        # 每张图用不同底色，避免被当成重复图片
        color = ((i * 7) % 256, (i * 13) % 256, (i * 29) % 256)
        im = Image.new("RGB", (96, 96), color)
        # 描一个随序号变化的方块，保证 dHash 不同
        for x in range((i * 3) % 48, (i * 3) % 48 + 20):
            for y in range(10, 30):
                im.putpixel((x % 96, y), (255 - color[0], 128, 40))
        exif = make_exif(34.797 + i * 0.0001, 114.307 + i * 0.0001)
        im.save(out / f"IMG_{i:04d}.jpg", quality=90, exif=exif)
    print(f"  生成 {category}: {count} 张")


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, *cmd],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="chrys_selftest_"))
    raw = tmp / "_raw"
    dataset = tmp / "dataset"
    print(f"临时目录：{tmp}")

    failures: list[str] = []

    try:
        print("\n[1/5] 生成合成原图")
        for cat in CATEGORIES:
            make_images(raw, cat, IMAGES_PER_CATEGORY)

        print("\n[2/5] 批量重命名")
        for cat in CATEGORIES:
            code, out = run(
                [
                    str(CODE_DIR / "prepare_dataset.py"),
                    "rename",
                    "--src",
                    str(raw / cat),
                    "--category",
                    cat,
                ],
                cwd=tmp,
            )
            if code != 0:
                failures.append(f"rename {cat}: {out}")
        sample = sorted(p.name for p in (raw / CATEGORIES[0]).iterdir())[:2]
        print(f"  重命名样例：{sample}")
        if not all(n.startswith(CATEGORIES[0]) for n in sample):
            failures.append(f"重命名格式不符：{sample}")

        print("\n[3/5] 按植株划分（8:1:1）")
        for cat in CATEGORIES:
            code, out = run(
                [
                    str(CODE_DIR / "prepare_dataset.py"),
                    "split",
                    "--src",
                    str(raw / cat),
                    "--category",
                    cat,
                    "--per-plant",
                    str(PER_PLANT),
                    "--dst",
                    str(dataset),
                ],
                cwd=tmp,
            )
            if code != 0:
                failures.append(f"split {cat}: {out}")
        # 检查划分结果
        total = 0
        for split in ("train", "val", "test"):
            n = sum(
                len([p for p in (dataset / split / c).iterdir() if p.suffix.lower() in IMAGE_EXTS])
                for c in CATEGORIES
                if (dataset / split / c).is_dir()
            )
            total += n
            print(f"  {split}: {n} 张")
        if total != IMAGES_PER_CATEGORY * len(CATEGORIES):
            failures.append(f"划分后总数 {total} != {IMAGES_PER_CATEGORY * len(CATEGORIES)}")

        print("\n[4/5] 生成 labels.csv")
        code, out = run(
            [
                str(CODE_DIR / "build_labels.py"),
                "--root",
                str(dataset),
                "--out",
                str(dataset / "labels.csv"),
            ],
            cwd=tmp,
        )
        print("  " + out.strip().replace("\n", "\n  ")[:600])
        if code != 0:
            failures.append(f"build_labels: {out}")
        csv_path = dataset / "labels.csv"
        if csv_path.is_file():
            lines = csv_path.read_text(encoding="utf-8-sig").strip().splitlines()
            print(f"  labels.csv 行数（含表头）：{len(lines)}")
            if len(lines) - 1 != total:
                failures.append(f"labels.csv 数据行 {len(lines)-1} != 图片数 {total}")

        print("\n[5/5] 数据集结构自查")
        code, out = run(
            [
                str(CODE_DIR / "check_dataset.py"),
                "--root",
                str(dataset),
                "--strict",
            ],
            cwd=tmp,
        )
        print("  " + out.strip().replace("\n", "\n  ")[:2500])
        if code != 0:
            failures.append("check_dataset --strict 返回非零码")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\n已清理临时目录 {tmp}")

    print("\n" + "=" * 60)
    if failures:
        print(f"[自测不通过] {len(failures)} 项：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[自测通过] 整条流水线可用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
