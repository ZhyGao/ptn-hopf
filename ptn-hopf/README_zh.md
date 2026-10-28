# 三叉神经痛：单个 PTN 被试的 Hopf 模拟示例

本项目提供一名 PTN 的真实处理后输入，以 **46 名 HLC 的组平均 FO/TPM**
作为健康参考。完整演示：**个体模型拟合 → 双侧脑区刺激 → 刺激热力图 → FO 变化**。

输入约 79 KiB，不包含原始 MRI、BOLD 时序、临床信息或其他人的个体数据。
LEiDA 聚类由外部流程完成，本项目读取其状态中心，并将模拟信号投影到相同状态空间。

本仓库提供示例输入、完整代码、安装方法和运行说明。运行后会在本机生成结果。

## 安装与运行

建议 Python 3.12，支持 Python 3.10 及以上；不需要 MATLAB 或 GPU。
在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py --validate-only
python run_demo.py --workers 4
```

Windows 的环境激活命令为 `.venv\Scripts\activate`。
若要使用本次验证环境的版本，可安装 `requirements-tested.txt`。

默认即完整配置，输出到 `outputs/full/`；不会减少重复次数或网格。也可显式运行：

```bash
python run_demo.py --config configs/full.json --workers 4 --output outputs/full
```

| 内容 | 完整配置（默认） |
|---|---|
| G 范围 | 0–0.5，步长 0.005，共 101 个 G |
| 每个 G 模拟次数 | 200 |
| 刺激 a 范围 | −0.4–0.4，步长 0.01，共 81 档 |
| 每个刺激条件模拟次数 | 50 |
| 双侧靶点 | 全部 45 对 |
| 模拟脑区 | AAL116 |

完整配置将产生3,645个刺激格点，合计模拟200,250条轨迹。
零刺激条件与靶点无关，因此每次重复只模拟一次，再复制到各靶点，不减少 n。
首次运行会编译 Numba 数值内核。`--workers` 表示并行进程数，可设为 1。
本次验证服务器使用 24 个进程，完整模拟约九分钟；个人电脑耗时会更长。

## 运行后生成的结果

运行后结果保存在`outputs/full/`。完成模拟后，可根据该目录的数值重新绘图：

```bash
python run_demo.py --plot-only --output outputs/full
```

仓库不附带已经跑好的结果，下面这些文件由代码运行后生成。

主要文件：

- `overview.png`：A为State 1 FO热图，B为TPM距离热图，C为G选择曲线，D为FO变化。
- `stimulation_fo_heatmap.png`：各靶点、各刺激强度下的平均State 1 FO。
- `stimulation_tpm_heatmap.png`：模拟平均TPM到HLC组平均TPM的Frobenius距离。
- `fit.png`：G搜索曲线和选中的G。
- `fo_change.png`：经验PTN、模拟基线、选定刺激、HLC组平均的FO，按State 1/State 2分组。
- `fit.csv`、`stimulation.csv`、`fo_comparison.csv`：拟合、刺激和FO变化的数值。
- 两个`*_matrix.csv`：完整精度的FO和TPM热图矩阵。
- `summary.json`：最优G、刺激条件及主要指标。
- `run_metadata.json`：输入和代码哈希、运行环境、参数、耗时和完成状态。
- `heatmap_display.json`：两个热图的p90色阶上限和四面板对应关系。

热图采用论文的p90显示范围，超过上限的值使用色图端点颜色。所有格子均着色，
数值CSV保留完整原始值。图中误差棒为不同噪声重复的标准差SD。
G拟合和刺激条件选择仍按方法中的对称KL标准计算。

实际使用的 **2×116 聚类中心**同时保存在 `data/ptn_demo.npz` 和
`data/state_centers.csv` 中；CSV 列名为 AAL 脑区，行名为 State 1/State 2。

## 方法口径

- **拟合选 G**：每次模拟先计算与经验 PTN FO 的对称 KL，再对重复求平均。
- **评价刺激**：先对重复的 FO 求平均，再计算与 HLC 组平均 FO 的对称 KL。
- **基线**：所有脑区 a=0。相同 G 下各刺激条件使用相同的噪声种子，降低比较中的随机差异。
- **刺激**：只改变指定的一对左右脑区的 a；其他脑区和个体 SC 保持原参数。
- **TPM**：作为辅助结果输出；本示例按 FO-KL 选择刺激条件。

这是单名被试的方法演示。选中条件和展示结果使用同一组重复，属于描述性结果，
不代表独立验证或临床疗效。输入来自此前已分析、数据完整且经验 State 1 FO 低于
HLC 均值的示例被试。

## 断点续跑与检查

```bash
python run_demo.py --config configs/full.json --workers 4 --output outputs/full --resume
python -m pip install pytest
python -m pytest -q
```

续跑会核对输入、配置、数值代码及环境版本；可改变并行进程数。
更换配置应使用新的输出目录。已有结果不会被静默覆盖。

更多细节见[数据字典](data/README.md)和[模型方法](docs/methods.md)。
LEiDA外部仓库的确切链接和版本待作者补充；
运行本项目不依赖该外部仓库。发布时由项目所有者确定代码与数据的许可条款。
