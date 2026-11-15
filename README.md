# 开封菊花视觉分类（Kaifeng Chrysanthemum Visual Classification）

基于 Ultralytics YOLO26 分类模型的细粒度菊花图像分类项目，数据集采集自开封菊花文化节展区。

> 本项目对应课程实践任务书《开封菊花视觉分类 · 数据集采集与构建任务书》。
> 当前状态：**框架搭建完成，等待外业采集与训练**。

---

## 目录

- [一、小组信息与分工](#一小组信息与分工)
- [二、数据集概述与类别说明](#二数据集概述与类别说明)
- [三、环境配置](#三环境配置)
- [四、训练过程](#四训练过程)
- [五、结果与分析](#五结果与分析)
- [六、局限与改进](#六局限与改进)
- [附录：项目结构与脚本用法](#附录项目结构与脚本用法)

---

## 一、小组信息与分工

> 状态：**待填写**。任务书 3.1 节要求分工落到具体动作，不能写"甲负责采集、乙负责代码"。

| 成员 | 学号 | 主要职责 | 具体交付物 |
| --- | --- | --- | --- |
| 甲 | 待填 | 类别方案与外业采集 | 类别清单、GPS 抽检记录、`_raw/` 原始库 |
| 乙 | 待填 | 数据处理与训练 | `labels.csv`、划分记录、训练脚本与日志 |
| 丙 | 待填 | 结果分析与文档 | 混淆矩阵分析、错误样本集、README 与汇报 PPT |

- 组名：待填
- 采集时间：待填
- 采集地点：待填（龙亭公园 / 清明上河园 / 铁塔公园 展区，具体点位见第二部分）

---

## 二、数据集概述与类别说明

> 状态：**已采集完成，480 张，10 类 × 48 张**。

### 2.1 概况

| 项目 | 内容 |
| --- | --- |
| 类别数 | 10 |
| 总张数 | 480（每类 48 张） |
| 采集时间 | 2026-11-12 16:23—17:50 |
| 采集地点 | 开封（铁塔公园展区），约 34.8159—34.8169°N，114.3625—114.3650°E |
| 拍摄设备 | iQOO 13（216 张）、iQOO Z10 Turbo（264 张） |
| 图像规格 | 长边 2048，JPG，质量 94，4:4:4 子采样 |
| 色彩空间 | sRGB（原图为 Display P3，已转换） |
| GPS 完整率 | 480/480 = 100% |
| 划分比例 | 训练 : 验证 : 测试 = 360 : 60 : 60（每类 36 / 6 / 6） |
| 植株数 | 每类 8 株，每株 6 张（对应 6 个拍摄位） |

### 2.2 类别说明

划分依据为**花色 + 瓣型/花型**的组合维度（任务书 4.1 节推荐做法）。两位序号保证
文件夹名字母序即序号序，从而 YOLO 的类别索引稳定可预判（任务书 11.2 节）。

| 序号 | 文件夹名 | 中文类别名 | 划分依据 | 样本数量 |
| --- | --- | --- | --- | --- |
| 01 | `01_huangse_pingban` | 黄色平瓣 | 花色=黄，瓣型=平瓣 | 48 |
| 02 | `02_baise_guanban` | 白色管瓣 | 花色=白，瓣型=管瓣 | 48 |
| 03 | `03_lvse_jiban` | 绿色畸瓣 | 花色=绿，瓣型=畸瓣 | 48 |
| 04 | `04_fense_pingban` | 粉色平瓣 | 花色=粉，瓣型=平瓣 | 48 |
| 05 | `05_hongse_qiuxing` | 红色球型 | 花色=红，花型=球型 | 48 |
| 06 | `06_fuse_feiwu` | 复色飞舞 | 花色=复色，花型=飞舞 | 48 |
| 07 | `07_huangse_guanban` | 黄色管瓣 | 花色=黄，瓣型=管瓣 | 48 |
| 08 | `08_fense_chiban` | 粉色匙瓣 | 花色=粉，瓣型=匙瓣 | 48 |
| 09 | `09_huangse_qiuxing` | 黄色球型 | 花色=黄，花型=球型 | 48 |
| 10 | `10_hongse_sheban` | 红色舌瓣 | 花色=红，瓣型=舌瓣 | 48 |

**与任务书示例方案的差异**：任务书 4.2 节给的是示例，允许按展区实际调整。
本组有 4 类不同——粉色平瓣（示例为粉色莲座）、粉色匙瓣（示例为白色平瓣）、
黄色球型（示例为紫色匙瓣）、红色舌瓣（示例为双色托桂）。原因是部分示例品种
在展区没有找到或形态难以稳定辨别，改用现场能可靠区分的类别。详见
[类别方案](docs/category_scheme.md)。

### 2.3 按植株与拍摄位的分布

每类 8 株 × 6 张 = 48 张。6 张对应任务书 5.3 节的 6 个拍摄位：

| 拍摄位 | 每株张数 | 每类小计 | 10 类合计 |
| --- | --- | --- | --- |
| 正面平视 | 1 | 8 | 80 |
| 45° 斜俯视 | 1 | 8 | 80 |
| 正俯视 | 1 | 8 | 80 |
| 侧面 90° | 1 | 8 | 80 |
| 近距特写 | 1 | 8 | 80 |
| 整株环境 | 1 | 8 | 80 |
| 合计 | 6 | 48 | 480 |

植株分布：每类 P01—P08 共 8 株，每株 6 张。`plant_id` 直接取自采集时的
植株目录（`_raw/{类别}/{植株}/`），不是靠序号推断。

### 2.4 图像预处理说明

原图为 3072×4096 / 4096×3072、Display P3、单张 2.6—7.9 MB，合计 2.1 GB。
按任务书要求做了四项处理（脚本见 [convert_photos.py](code/convert_photos.py)）：

| 处理 | 原因 | 结果 |
| --- | --- | --- |
| Display P3 → sRGB | 训练库（PIL/torchvision）不读 ICC，会把 P3 数值当 sRGB 用，颜色失真 | 480 张全部转换 |
| 横屏 → 竖屏 | 98 张横屏集中在 6 个类别（4 个类别一张都没有），模型会学到"横屏=某类"的伪特征 | 98 张旋转 90° |
| 绿色畸瓣居中裁 1/2 | 该类绿色花头平均只占画面 8%，周围全是粉/紫/黄其他品种，模型会学到"粉色背景=绿色畸瓣" | 48 张裁剪，占比升到 20% |
| 长边缩到 2048 | 任务书要求 1024—2048；原图 4080—4096 过大 | 全部 2048 |

**关于尺寸**：绿色畸瓣裁 1/2 后长边正好从 4096 降到 2048，不需要再缩放——
少一次重采样，画质更好。其余类别直接缩放到 2048。

**关于 HDR**：已检查全部 480 张，均为 SOF0 baseline、8-bit/通道、3 分量，
无 Ultra HDR 增益图（APP2 无 21496/MPF 标记），无 PQ/HLG 传输函数。
**没有 HDR 图片**，因此不需要色调映射。

**关于压缩**：质量 94 + 4:4:4 子采样（不做色度降采样），PSNR 45.6 dB，
480 张共 438 MB，在任务书 500 MB 上限内。质量 95 会到 488 MB，加上权重与
图表会逼近上限；质量 90 只有 327 MB 但 PSNR 掉到 43.9 dB，花瓣边缘的细微
色差开始丢失。EXIF（GPS、拍摄时间、设备型号）与 ICC 全部保留。

**关于横屏**：98 张横屏集中在每株的第 4—6 张，对应"侧面 90°、近距特写、
整株环境"三个拍摄位——拍侧面时自然会横握手机。旋转后花头方向与竖屏图
不一致（侧躺），但分类任务对旋转不敏感，训练时还有 `RandomHorizontalFlip`。

---

## 三、环境配置

> 状态：**配置步骤已定，`yolo checks` 截图待补充**。

### 3.1 版本要求

| 项目 | 要求 / 本机实测 |
| --- | --- |
| 操作系统 | Windows 11（10.0.26100） |
| Python | 3.11 或 3.12 推荐（本机 3.14.7 亦可，ultralytics 8.4.210 声明支持至 3.15） |
| PyTorch | 2.15.0（GPU 版需按 CUDA 版本选 index-url） |
| torchvision | 0.29.2 |
| Ultralytics | 8.4.210 |
| 显卡 | 待填（无独显时 `device: "cpu"`，训练约 40—60 分钟） |

### 3.2 创建虚拟环境

```powershell
cd C:\Users\Administrator\Documents\kaifeng-chrysanthemum-visual-classification-model
python -m venv .venv
.venv\Scripts\activate
```

若 PowerShell 提示"无法加载文件 activate.ps1"：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 3.3 安装依赖（顺序不可颠倒）

```powershell
# 有 NVIDIA 显卡
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 只有 CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 再装其余依赖
pip install -r code/requirements.txt
```

### 3.4 环境变量

本项目的 `code/common.py` 会在导入 ultralytics 前自动设置 `YOLO_CONFIG_DIR`，
指向仓库内的 `.yolo_config/`，因此**脚本方式运行无需手工设置**。

命令行方式（`yolo` 命令）运行训练时需要手工设置：

```powershell
$env:YOLO_CONFIG_DIR = "C:\Users\Administrator\Documents\kaifeng-chrysanthemum-visual-classification-model\.yolo_config"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:OMP_NUM_THREADS = "4"
yolo checks
```

### 3.5 环境自检

```powershell
python code/check_env.py
```

预期输出（片段）：

```
Python 版本 : 3.14.7
PyTorch     : 2.15.0
CUDA 可用   : False
Ultralytics : 8.4.210
Pillow      : 13.0.0
YOLO_CONFIG_DIR     : ...\.yolo_config
```

> **待补充**：`yolo checks` 输出截图（任务书自检清单第 20 项）。

---

## 四、训练过程

> 状态：**待训练**。

### 4.1 模型型号与选择理由

| 项目 | 本组取值 |
| --- | --- |
| 模型型号 | `yolo26n-cls.pt`（2.8M 参数） |
| 输入尺寸 | 224 |
| 微调策略 | 全量微调（`freeze=None`） |
| 增强策略 | 默认 randaugment |
| 优化器与学习率 | `auto`（框架自选，`lr0` 被忽略） |
| 训练轮数 | 50（`patience=15` 早停） |

选择理由：样本量 480 张属小数据集，n 规模足以拟合，训练最快；224 是
ImageNet 预训练尺寸，省时且与迁移学习匹配。若与别组撞车，优先改
`model` 与 `imgsz`——这两项对指纹影响最直接（任务书 3.3 节要求至少三项不同）。

### 4.2 完整超参表

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| model | `yolo26n-cls.pt` | 网络规模 n/s/m |
| imgsz | 224 | 输入分辨率 |
| batch | 32 | 显存不足时减半 |
| epochs | 50 | 配合 patience 早停 |
| optimizer | auto | 手动控学习率需显式指定 |
| lr0 | 0.001 | `optimizer=auto` 时被忽略 |
| dropout | 0.0 | 过拟合时设 0.2 |
| freeze | None | 全量微调 |
| seed | 42 | 保证可复现 |
| patience | 15 | 早停耐心值 |
| workers | 2 | Windows 建议 0—4 |
| device | 0（无显卡自动切 cpu） | — |
| cache | False | 数据集小可设 True |

### 4.3 训练命令

```powershell
# 先跑 5 轮验证流程是否通（任务书阶段 6 第 2 步）
python code/train.py --epochs 5

# 确认无误后正式训练，并保存控制台日志
python code/train.py 2>&1 | Tee-Object -FilePath results\train_log.txt -Encoding utf8
```

### 4.4 训练耗时

| 项目 | 内容 |
| --- | --- |
| 设备 | 待填 |
| 总耗时 | 待填 |
| 每轮耗时 | 待填 |

### 4.5 模型指纹

| 项目 | 本组取值 |
| --- | --- |
| 模型型号 | `yolo26n-cls` |
| 输入尺寸 | 224 |
| 微调策略（freeze 层数） | None（全量） |
| 增强策略 | randaugment |
| 优化器与学习率 | auto |
| 训练轮数 | 50 |
| 模型指纹（8 位哈希） | 待训练后由 `train.py` / `evaluate.py` 输出 |

> 指纹由 `code/common.py:model_fingerprint()` 对六项配置排序后取 MD5 前 8 位。

### 4.6 训练曲线

> **待补充**：`results/runs/chrys_yolo26n/results.png` 及解读。

---

## 五、结果与分析

> 状态：**待训练**。

### 5.1 三集合准确率

| 集合 | 用途 | top1 | top5 |
| --- | --- | --- | --- |
| train | 训练时见过 | 待填 | 待填 |
| val | 选最优轮次 | 待填 | 待填 |
| test | 最终验收（只看一次） | 待填 | 待填 |

原始数据见 `results/eval_summary.json`，生成命令：

```powershell
python code/evaluate.py
```

### 5.2 混淆矩阵

> **待补充**：`confusion_matrix.png`、`confusion_matrix_normalized.png` 及解读。

必须做的一件事（任务书 12.2 节）：把混淆最严重的那一对类别各挑 5 张原图
并列对比，写进本节。这是分析报告里最有价值的部分。

### 5.3 错误样本分析

| 编号 | 文件名 | 真实类别 | 预测类别 | 原因分类 |
| --- | --- | --- | --- | --- |
| 待填 | | | | 标注错误 / 定义模糊 / 拍摄质量 / 真实困难 |

错误样本原图放 `results/error_cases/`。

### 5.4 组间差异化核查

> 待填：与哪些组做过比对、相似度与重复率结果、截图。

```powershell
python code/compare_datasets.py D:\path\to\other_group\train
```

---

## 六、局限与改进

> 状态：**待采集与训练后如实填写**。本部分不扣分，能识别自己方案的短板是
> 工程素养的体现（任务书 15.3 节）。

需要如实回答的四个问题：

1. **哪类样本偏少**：待填。
2. **哪类背景偏单一**：待填。
3. **哪组配置未做对比**：待填（建议做一组：224 vs 320，或全量 vs 冻结前 4 层）。
4. **如果重做会怎么改**：待填。

已知的结构性局限（框架阶段即可预见）：

- 每类 48 张属小样本，细粒度分类对形态相近的类别（如 `07_huangse_guanban`
  与 `01_huangse_pingban`）容易混淆；
- 单一展区、单一时段拍摄，模型可能学到展区背景特征而非花的特征；
- 10 类样本较少时 top5 准确率会很快接近 1.0，参考价值有限，主指标应看 top1；
- **绿色畸瓣构图偏弱**：该类绿色花头原本平均只占画面 8%，已通过居中裁 1/2
  提升到 20%，但仍有 8/48 张低于 10%。这是采集时的构图问题，裁剪只能缓解
  不能根治——如果重做，应靠近拍摄让花头占画面 1/2 以上。

---

## 附录：项目结构与脚本用法

### A.1 目录结构

```
kaifeng-chrysanthemum-visual-classification-model/
├── train/                       # 训练集（10 个类别文件夹）
├── val/                         # 验证集（结构同 train）
├── test/                        # 测试集（结构同 train）
├── labels.csv                   # 元数据表（由 build_labels.py 生成）
├── README.md                    # 本文档
├── code/                        # 全部脚本
│   ├── common.py                # 公共路径、类别枚举、模型指纹
│   ├── train.py                 # 训练（顶部 CONFIG 字典）
│   ├── evaluate.py              # 三集合评估 + eval_summary.json
│   ├── predict.py               # 单张/批量预测
│   ├── build_labels.py          # 从 EXIF 生成 labels.csv
│   ├── check_dataset.py         # 数据集结构自查
│   ├── check_env.py             # 环境自检
│   ├── compare_datasets.py      # 组间 dHash 相似度核查
│   ├── prepare_dataset.py       # 重命名 / 去重 / 按植株划分
│   ├── convert_photos.py        # P3→sRGB、横屏转竖屏、缩放重编码
│   ├── selftest.py              # 用合成图片跑通全流程的开发自测
│   └── requirements.txt         # 固定版本依赖
├── results/                     # 训练日志、图表、评估结果
│   ├── runs/chrys_yolo26n/      # 训练产物（weights/、results.png 等）
│   ├── eval_summary.json        # 三集合准确率
│   ├── train_log.txt            # 控制台输出
│   └── error_cases/             # 错误样本原图
├── docs/
│   ├── category_scheme.md       # 类别方案（待现场核对）
│   ├── collection_plan.md       # 采集方案与记录表
│   └── submission_checklist.md  # 提交前 28 项自检清单
├── _raw/                        # 原始照片库（不入库，提交前删除）
└── 开封菊花视觉分类_数据集采集任务书.pdf   # 课程任务书（参考，不进压缩包）
```

### A.2 常用命令

| 场景 | 命令 |
| --- | --- |
| 环境自检 | `python code/check_env.py` |
| 原图转换（P3→sRGB、横屏转竖屏、缩放） | `python code/convert_photos.py --src <原图目录>` |
| 按植株目录重命名 | `python code/prepare_dataset.py rename-plants --src _raw` |
| 按植株目录划分 | `python code/prepare_dataset.py split-plants --src _raw` |
| 生成元数据表 | `python code/build_labels.py` |
| 数据集自查 | `python code/check_dataset.py --strict` |
| 训练 | `python code/train.py` |
| 评估 | `python code/evaluate.py` |
| 预测 | `python code/predict.py test/01_huangse_pingban` |
| 组间相似度 | `python code/compare_datasets.py <本组> <他组>` |
| 开发自测 | `python code/selftest.py` |

### A.2.1 从原图重建数据集的完整流程

```powershell
# 1. 转换：P3→sRGB、横屏转竖屏、长边 2048、质量 94
python code/convert_photos.py --src "C:\Users\Administrator\Downloads\菊花"

# 2. 重命名：{类别}_{三位序号}.jpg，解决跨植株重名
python code/prepare_dataset.py rename-plants --src _raw

# 3. 划分：整株分配，每类 6 株 train / 1 株 val / 1 株 test
python code/prepare_dataset.py split-plants --src _raw

# 4. 生成元数据表（plant_id 从 _raw 的植株目录读取）
python code/build_labels.py

# 5. 自查
python code/check_dataset.py --strict
```

### A.3 关于版本库与交付包的区别

仓库和提交用的 ZIP 不是一回事，别混淆：

| | 内容 | 说明 |
| --- | --- | --- |
| **git 仓库** | 代码、文档、目录骨架 | `.gitignore` 排除了图片、`*.pt` 权重、`.venv/`、`_raw/`。克隆下来只有骨架，需要自己放图、自己训练 |
| **提交 ZIP** | 七项，含图片与 `best.pt` | 从**本地工作目录**打包，不是从 git 克隆目录打包 |

因此打包前要确认本地确实有：
`train/ val/ test/` 里的图片、`labels.csv`、`results/runs/chrys_yolo26n/weights/best.pt`
以及四张图表。缺 `best.pt` 时按任务书 15.2 节单独上传并在 README 给出下载链接。

### A.4 提交前打包

任务书阶段 7 第 3 条：压缩包内**不要包含上级文件夹**，只选中七项：

```
train/  val/  test/  labels.csv  README.md  code/  results/
```

在仓库根目录执行：

```powershell
python code/check_dataset.py --strict
# 确认通过后，选中七项压缩（PowerShell 示例）
Compress-Archive -Path train,val,test,labels.csv,README.md,code,results `
  -DestinationPath "2025001_张三_菊花数据集_10类_480张.zip"
```

打包前请核对 `docs/submission_checklist.md` 中的 28 项自检清单。
