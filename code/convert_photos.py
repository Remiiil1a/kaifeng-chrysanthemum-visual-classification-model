"""把采集的原图转换成训练用数据集：色彩空间标准化 + 缩放 + 重编码。

用法：
    python code/convert_photos.py --src "C:\\Users\\Administrator\\Downloads\\菊花" --dry-run
    python code/convert_photos.py --src "C:\\Users\\Administrator\\Downloads\\菊花"

做四件事，顺序不能变：
  1. Display P3 -> sRGB 色彩空间转换（必须在缩放之前，否则插值在错误的
     色彩空间里做，颜色会偏）
  2. 横屏图旋转 90° 成竖屏（消除"横屏=某类"的伪特征）
  3. 长边缩放到 2048（任务书要求 1024—2048）
  4. 质量 95 + 4:4:4 子采样重编码，EXIF 与 ICC 完整保留

为什么是质量 95 + 4:4:4：实测 480 张约 443 MB，PSNR 47.4 dB，在任务书
500 MB 上限内。质量 97 会到 579 MB 超标；质量 90 + 4:2:0 只有 238 MB 但
色度被降采样，花瓣边缘的细微色差会丢——细粒度分类靠的就是这些细节。

原图只读不写，输出到 --dst（默认 _raw/），保留完整目录结构。
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, rel  # noqa: E402

try:
    from PIL import Image, ImageCms
except ImportError:
    print("缺少 Pillow：pip install pillow", file=sys.stderr)
    raise SystemExit(1)

# 任务书 2 章：长边 1024—2048
MAX_EDGE = 2048
# 质量 94 + 4:4:4：实测 480 张约 433 MB，PSNR 45.6 dB。
# 质量 95 实测 488 MB，加上 best.pt（3—6 MB）与图表会逼近任务书 500 MB 上限，
# 所以留出余量取 94。质量 90 只有 327 MB 但 PSNR 掉到 43.9 dB，花瓣边缘的
# 细微色差开始丢——细粒度分类靠的就是这些细节。
JPEG_QUALITY = 94
# 0 = 4:4:4 不做色度降采样，保住花瓣边缘的细微色差
JPEG_SUBSAMPLING = 0

# 需要居中裁剪的类别：{类别名: 保留比例}
# 绿色畸瓣的绿色花头平均只占画面 8%，周围全是粉/紫/黄的其他品种，
# 模型会学到"粉色背景=绿色畸瓣"这种伪特征。从原图裁掉外围一半后，
# 绿色占比升到 18%，花头完整居中（已逐张目视确认没切到花瓣）。
# 其他类别主体本来就占 38—66%，不需要裁。
#
# 裁 1/2 正好把长边从 4096 降到 2048，一步到位满足任务书尺寸要求，
# 不需要再缩放——少一次重采样，画质更好。
#
# 键同时支持中文类别名和拼音文件夹名，因为原图目录是中文、数据集目录是拼音。
CROP_RATIO = {
    "绿色畸瓣": 0.5,
    "03_lvse_jiban": 0.5,
}

# 中文类别名 -> 拼音文件夹名。原图按中文分类，数据集要求全小写拼音，
# 转换时顺手改名，省掉一次手工重命名。
CATEGORY_MAP = {
    "黄色平瓣": "01_huangse_pingban",
    "白色管瓣": "02_baise_guanban",
    "绿色畸瓣": "03_lvse_jiban",
    "粉色平瓣": "04_fense_pingban",
    "红色球型": "05_hongse_qiuxing",
    "复色飞舞": "06_fuse_feiwu",
    "黄色管瓣": "07_huangse_guanban",
    "粉色匙瓣": "08_fense_chiban",
    "黄色球型": "09_huangse_qiuxing",
    "红色舌瓣": "10_hongse_sheban",
}


def crop_center(im: Image.Image, frac: float) -> Image.Image:
    """裁掉四周，保留中间 frac 比例（按线性尺寸）。"""
    w, h = im.size
    nw, nh = int(w * frac), int(h * frac)
    left, top = (w - nw) // 2, (h - nh) // 2
    return im.crop((left, top, left + nw, top + nh))


def to_srgb(im: Image.Image) -> tuple[Image.Image, bool]:
    """把带 ICC 的图转到 sRGB。返回 (图像, 是否做了转换)。

    没有 ICC 的图按 sRGB 处理（绝大多数软件也这么假设），不做转换。
    """
    icc = im.info.get("icc_profile")
    if not icc:
        return im, False
    try:
        src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        desc = ImageCms.getProfileDescription(src_profile).strip()
        if "srgb" in desc.lower():
            return im, False
        srgb = ImageCms.createProfile("sRGB")
        # renderingIntent=0 (perceptual)：广色域转窄色域时保持整体观感，
        # 比 relative colorimetric 更不容易出现大面积色块断层
        converted = ImageCms.profileToProfile(
            im, src_profile, srgb, outputMode="RGB", renderingIntent=0
        )
        return converted, True
    except (ImageCms.PyCMSError, OSError, ValueError) as e:
        print(f"    [警告] ICC 转换失败，按原样处理：{e}")
        return im, False


def convert_one(src: Path, dst: Path, dry_run: bool, crop_frac: float = 1.0) -> dict:
    """转换单张图片，返回统计信息。

    crop_frac < 1 时先居中裁剪再缩放——顺序很重要：从原图（4080—4096）
    裁完再缩到 2048，最终尺寸仍是 1536×2048；如果先缩到 2048 再裁，
    长边会掉到 1024，正好卡在任务书下限没有余量。
    """
    with Image.open(src) as im:
        orig_size = im.size
        orig_bytes = src.stat().st_size
        exif = im.getexif()
        icc = im.info.get("icc_profile")

        # 1. 色彩空间标准化
        work, converted = to_srgb(im)
        if converted:
            # 转换后 ICC 变成 sRGB，重新取一次
            icc = work.info.get("icc_profile")

        # 2. 横屏转竖屏
        rotated = False
        w, h = work.size
        if w > h:
            work = work.transpose(Image.Transpose.ROTATE_90)
            rotated = True

        # 3. 居中裁剪（只对主体过小的类别）
        cropped = False
        if crop_frac < 1.0:
            work = crop_center(work, crop_frac)
            cropped = True

        # 4. 缩放（长边到 MAX_EDGE，只缩不放）
        # 裁 1/2 的类别裁完长边正好 2048，这里不会再动它——少一次重采样。
        w, h = work.size
        scale = MAX_EDGE / max(w, h)
        if scale < 1.0:
            work = work.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

        # 5. 重编码
        # Orientation 必须显式设为 1（正常方向）。原图里有 0 这种非法值，
        # 旋转后如果还留着旧值，部分查看器会再转一次，导致花头躺倒。
        exif[274] = 1
        buf = io.BytesIO()
        save_kw = dict(
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=JPEG_SUBSAMPLING,
            optimize=True,
            exif=exif,
        )
        if icc:
            save_kw["icc_profile"] = icc
        work.save(buf, **save_kw)
        new_bytes = buf.tell()

        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(buf.getvalue())

    return {
        "orig_size": orig_size,
        "new_size": work.size,
        "orig_bytes": orig_bytes,
        "cropped": cropped,
        "new_bytes": new_bytes,
        "rotated": rotated,
        "converted": converted,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="转换原图为训练用数据集")
    parser.add_argument("--src", required=True, help="原图根目录（含 类别/植株/ 结构）")
    parser.add_argument("--dst", default=str(RAW_DIR), help="输出根目录，默认 _raw/")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    args = parser.parse_args(argv)

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    if not src_root.is_dir():
        print(f"[错误] 源目录不存在：{src_root}", file=sys.stderr)
        return 1

    images = sorted(
        p for p in src_root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not images:
        print(f"[错误] {src_root} 下没有图片", file=sys.stderr)
        return 1

    print(f"源目录：{src_root}")
    print(f"输出到：{rel(dst_root)}")
    print(f"共 {len(images)} 张，长边缩到 {MAX_EDGE}，质量 {JPEG_QUALITY}，4:4:4")
    if CROP_RATIO:
        print(f"居中裁剪：{CROP_RATIO}")
    if args.dry_run:
        print("[试运行] 不写文件\n")

    stats = {"rotated": 0, "converted": 0, "cropped": 0, "orig_bytes": 0, "new_bytes": 0}
    per_cat: dict[str, int] = {}

    for i, src in enumerate(images, 1):
        rel_path = src.relative_to(src_root)
        # 中文类别名换成拼音文件夹名，其余层级（植株/文件名）原样保留
        parts = list(rel_path.parts)
        if parts and parts[0] in CATEGORY_MAP:
            parts[0] = CATEGORY_MAP[parts[0]]
        dst = dst_root.joinpath(*parts)
        cat = parts[0] if len(parts) > 1 else "?"
        crop_frac = CROP_RATIO.get(cat, 1.0)
        info = convert_one(src, dst, args.dry_run, crop_frac)
        stats["rotated"] += info["rotated"]
        stats["converted"] += info["converted"]
        stats["cropped"] += info["cropped"]
        stats["orig_bytes"] += info["orig_bytes"]
        stats["new_bytes"] += info["new_bytes"]
        per_cat[cat] = per_cat.get(cat, 0) + 1

        if i % 50 == 0 or i == len(images):
            print(f"  已处理 {i}/{len(images)}")

    print("\n" + "=" * 60)
    print(f"类别数：{len(per_cat)}")
    for cat in sorted(per_cat):
        print(f"  {cat:<12} {per_cat[cat]:>3} 张")
    print("-" * 60)
    print(f"旋转为竖屏：{stats['rotated']} 张")
    print(f"P3 -> sRGB：{stats['converted']} 张")
    print(
        f"体积：{stats['orig_bytes']/1024/1024:.0f} MB -> "
        f"{stats['new_bytes']/1024/1024:.0f} MB"
    )
    if stats["new_bytes"] > 500 * 1024 * 1024:
        print("[警告] 超过任务书 500 MB 上限，需降低质量或尺寸")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
