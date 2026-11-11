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

> 状态：**类别方案已定框架，采集后补充分布统计**。

### 2.1 概况

| 项目 | 内容 |
| --- | --- |
| 类别数 | 10 |
| 总张数 | 待采集（目标 450 张，每类 45 张） |
| 采集时间 | 待填 |
| 采集地点 | 待填（开封市，约 34.7—34.9°N，114.2—114.4°E） |
| 拍摄设备 | 待填（手机后置主摄，关闭美颜/水印/滤镜） |
| 划分比例 | 训练 : 验证 : 测试 = 8 : 1 : 1（每类 36 / 4 / 5 张） |

### 2.2 类别说明

划分依据为**花色 + 瓣型**的组合维度（任务书 4.1 节推荐做法）。两位序号保证
文件夹名字母序即序号序，从而 YOLO 的类别索引稳定可预判（任务书 11.2 节）。

| 序号 | 文件夹名 | 中文类别名 | 划分依据 | 样本数量 |
| --- | --- | --- | --- | --- |
| 01 | `01_huangse_pingban` | 黄色平瓣 | 花色=黄，瓣型=平瓣 | 待采集 |
| 02 | `02_baise_guanban` | 白色管瓣 | 花色=白，瓣型=管瓣 | 待采集 |
| 03 | `03_lvse_jiban` | 绿色畸瓣 | 花色=绿，瓣型=畸瓣 | 待采集 |
| 04 | `04_fense_lianzuo` | 粉色莲座 | 花色=粉，花型=莲座 | 待采集 |
| 05 | `05_hongse_qiuxing` | 红色球型 | 花色=红，花型=球型 | 待采集 |
| 06 | `06_fuse_feiwu` | 复色飞舞 | 花色=复色，花型=飞舞 | 待采集 |
| 07 | `07_huangse_guanban` | 黄色管瓣 | 花色=黄，瓣型=管瓣 | 待采集 |
| 08 | `08_baise_pingban` | 白色平瓣 | 花色=白，瓣型=平瓣 | 待采集 |
| 09 | `09_zise_chiban` | 紫色匙瓣 | 花色=紫，瓣型=匙瓣 | 待采集 |
| 10 | `10_shuangse_tuogui` | 双色托桂 | 花色=双色，花型=托桂 | 待采集 |

**注意**：上表为任务书 4.2 节给出的示例方案。实际类别必须以展区标牌为准，
采集前逐类核对、必要时替换（任务书：不得凭印象编造品种名）。替换后需同步
修改：三个子集下的文件夹名、本表、以及已生成的 `labels.csv`。

### 2.3 按植株与拍摄位的分布

> 状态：**待采集**。任务书 5.3 节要求每类覆盖 6 个拍摄位、3—5 株不同植株。

| 拍摄位 | 每类目标张数 | 小计（10 类） |
| --- | --- | --- |
| 正面平视 | 8 | 80 |
| 45° 斜俯视 | 10 | 100 |
| 正俯视 | 6 | 60 |
| 侧面 90° | 6 | 60 |
| 近距特写 | 8 | 80 |
| 整株环境 | 7 | 70 |
| 合计 | 45 | 450 |

植株分布：每类建议 3—5 株，每株 9—15 张；`_raw/` 下按 `P01`、`P02` 编号
登记，划分时以植株为单位整体分配集合（见 `code/prepare_dataset.py split`）。

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

选择理由：样本量 450 张属小数据集，n 规模足以拟合，训练最快；224 是
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

- 每类 45 张属小样本，细粒度分类对形态相近的类别（如 `07_huangse_guanban`
  与 `01_huangse_pingban`）容易混淆；
- 单一展区、单一时段拍摄，模型可能学到展区背景特征而非花的特征；
- 10 类样本较少时 top5 准确率会很快接近 1.0，参考价值有限，主指标应看 top1。

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
| 批量重命名原图 | `python code/prepare_dataset.py rename --src _raw/01_huangse_pingban` |
| 去重 | `python code/prepare_dataset.py dedupe --src _raw/01_huangse_pingban` |
| 按植株划分 | `python code/prepare_dataset.py split --src _raw/01_huangse_pingban --category 01_huangse_pingban --per-plant 9` |
| 生成元数据表 | `python code/build_labels.py` |
| 数据集自查 | `python code/check_dataset.py --strict` |
| 训练 | `python code/train.py` |
| 评估 | `python code/evaluate.py` |
| 预测 | `python code/predict.py test/01_huangse_pingban` |
| 组间相似度 | `python code/compare_datasets.py <本组> <他组>` |
| 开发自测 | `python code/selftest.py` |

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
  -DestinationPath "2025001_张三_菊花数据集_10类_450张.zip"
```

打包前请核对 `docs/submission_checklist.md` 中的 28 项自检清单。
