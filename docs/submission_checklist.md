# 提交前自检清单

对应任务书第十三章 28 项。全部为"是"才可提交。打印本页逐项打勾。

| 序号 | 检查项 | 自动化 | 是/否 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 已按 3 人一组完成，分工表已写入 README | 手工 | | README 第一部分 |
| 2 | 类别数 ≥ 10，且各类形态差异可辨 | 半自动 | | `check_dataset.py` 报类别数 |
| 3 | 每类 ≥ 30 张，测试集每类 ≥ 5 张 | 自动 | | `check_dataset.py` |
| 4 | 全部为本组在开封现场拍摄，无网络图片 | 手工 | | 保留 `_raw/` 原图备查 |
| 5 | 每张图片均含 GPS 经纬度，抽检 10 张全部通过 | 自动 | | `build_labels.py` 报缺 GPS 清单 |
| 6 | labels.csv 行数 = 图片总张数 | 自动 | | `check_dataset.py` |
| 7 | labels.csv 中经纬度与图片 EXIF 一致 | 自动 | | 由 EXIF 直接生成，天然一致 |
| 8 | 经纬度落在开封范围（34.7—34.9°N，114.2—114.4°E） | 自动 | | `check_dataset.py` |
| 9 | 压缩包解压后直接是 train/val/test，无多余嵌套 | 手工 | | 只选七项压缩 |
| 10 | 三个子集均含全部 10 个类别文件夹 | 自动 | | `check_dataset.py` |
| 11 | 三个子集的类别文件夹名完全一致 | 自动 | | `check_dataset.py` |
| 12 | 图片直接位于类别文件夹下，无中间层 | 自动 | | `check_dataset.py` |
| 13 | 文件名全局唯一，无重名、无中文、无空格 | 自动 | | `check_dataset.py` |
| 14 | 同一植株的照片未跨集合（无数据泄漏） | 自动 | | `check_dataset.py` |
| 15 | 组间数据集相似度已自查，结果 < 70% | 手工 | | `compare_datasets.py` |
| 16 | train.py、evaluate.py、predict.py 三个脚本齐全 | 自动 | | 已具备 |
| 17 | 训练脚本有 `if __name__ == "__main__":` 保护块 | 自动 | | 已具备 |
| 18 | train.py 顶部有 CONFIG 字典，超参集中可改 | 自动 | | 已具备 |
| 19 | 环境变量设置方式已写入 README（含具体命令） | 手工 | | README 第 3.4 节 |
| 20 | yolo checks 的输出已截图保存 | 手工 | | 待补充 |
| 21 | 训练日志（控制台输出）已保存为 train_log.txt | 手工 | | README 第 4.3 节命令 |
| 22 | 四张图表齐全：results.png、confusion_matrix.png、confusion_matrix_normalized.png、val_batch0_pred.jpg | 手工 | | 训练时 `plots=True` 自动生成 |
| 23 | 三个集合的 top1 准确率均已记录到 eval_summary.json | 自动 | | `evaluate.py` |
| 24 | 混淆矩阵中易混的两类已做原图对比分析 | 手工 | | README 第 5.2 节 |
| 25 | 模型指纹已计算并登记，与同班其他组不同 | 半自动 | | `train.py` / `evaluate.py` 输出 |
| 26 | 在另一台电脑上按 README 重跑，能复现结果 | 手工 | | 建议组间互验 |
| 27 | README.md 含分工、类别、环境、训练、结果、局限六部分 | 手工 | | 已具备框架 |
| 28 | 压缩包命名符合第十五章规范 | 手工 | | `{学号}_{姓名}_菊花数据集_{N}类_{M}张.zip` |

## 一键自查

```powershell
python code/check_env.py
python code/check_dataset.py --strict
```

`--strict` 模式下有任一错误即以非零码退出，便于脚本化把关。

## 提交前 5 分钟必查

1. `_raw/` 目录已删除（或至少不放进压缩包）；
2. `results/` 里没有 `last.pt` 以外的临时文件；
3. `code/` 里没有 `__pycache__`；
4. `labels.csv` 用 Excel 打开无乱码（utf-8-sig）；
5. 换一台机器解压后，`python code/check_dataset.py` 能跑通。
