# 纠错码解码挑战赛 · POLAR(64,32) 统计列表译码方案

[![CI](https://github.com/Alwayshere-su/iflytek-ecc-decode/actions/workflows/ci.yml/badge.svg)](https://github.com/Alwayshere-su/iflytek-ecc-decode/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Docs: CC BY 4.0](https://img.shields.io/badge/docs-CC%20BY%204.0-lightgrey.svg)](LICENSE-docs)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)

> 2026 讯飞纠错码解码挑战赛参赛作品。在 AWGN 信道下，从 64 维含噪接收信号
> `y_0..y_63` 恢复 POLAR(64,32) 码字 `bit_0..bit_63`。
>
> 最终方案不是端到端神经网络，而是 **OSD 候选 + scale-mixture bit-MAP 列表译码**：
> 排行榜 BER **0.00675**，初赛 **Top 14 / 1841（前 0.8%）**。

## 🏆 赛事成绩

| 项目 | 结果 |
|---|---|
| 赛事 | 2026 讯飞开发者大赛 · 纠错码解码挑战赛 |
| 主办方 | 合肥工业大学 |
| 赛事规模 | 1841 支队伍 |
| 初赛排名 | **Top 14 / 1841（前 0.8%）** |
| 最佳 BER | **0.00675**（比特错误率，越低越好） |
| 提交次数 | 12 次 |
| 最终方案 | OSD order-3/order-4 候选 + scale-mixture bit-MAP |
| 本地复核 | order-4 在 10 万行验证集上 BER **0.006750** |

## 📖 项目背景

在 5G/6G 通信、卫星通信、深空探测和数据存储中，纠错码是保障信息可靠传输的核心技术。
传统解码算法（SC、BP、OSD 等）往往面临性能与复杂度的权衡。赛题鼓励使用神经网络
逼近最优解码边界，并提供 6400 万条训练样本。

本项目系统评估了神经网络路线，包括端到端 MLP/Transformer、Syndrome-NN、候选重排
和信道状态估计 NN。实验结论是：在本赛题的 (64,32) 线性码约束下，NN 很难从有限样本
中隐式学到 GF(2) 结构；候选重排虽然能提高 AUC，但集成进 bit-MAP 后 BER 反而变差。
最终交付的是一条**可解释、可复现、无需 GPU 训练**的统计解码链路。

## 🎯 赛事任务

- **编码方案**：POLAR(64, 32)，码长 64，信息位 32
- **信道模型**：加性高斯白噪声（AWGN），逐样本噪声功率存在混合
- **输入**：64 个浮点特征 `y_0..y_63`
- **输出**：64 个二进制比特 `bit_0..bit_63`
- **训练数据**：64,000,000 条样本，500 个 shard，每个 shard 128,000 对样本
- **评估指标**：BER（Bit Error Rate），越低越好

## 🛠️ 技术方案

### 1. 候选生成：OSD order-3 / order-4

按 `|y|` 的可靠性对 64 个比特排序，构造最可靠信息基（MRB），枚举 MRB 前 32 个
低可靠位的翻转组合：

| 候选集 | 每样本候选数 | 用途 |
|---|---:|---|
| OSD order-3 | 5,489 | 主候选集；诊断样本中真码字 1000/1000 在候选内 |
| OSD order-4 | +35,960（共 41,449） | 独立复核：10 万行比 order-3 少 51 bit |
| OSD order-5 | +201,376 | 未采用：远候选累积偏差，反而扰动后验 |

候选生成使用向量化 XOR + 多进程，10 万行 order-3 约 14 分钟（24 核 CPU）；
order-4 的候选数约为 order-3 的 7.5 倍，耗时和内存相应增加。

### 2. 似然：scale-mixture 边缘化逐样本噪声功率

信道不是单一 SNR，而是逐样本噪声功率 `s²` 变化（cv≈0.55）。正确的似然不是
“用一个全局 σ²”，而是对 `s²` 做边缘化：

```text
logw_c = logsumexp_s2 [ -32·log(s²) - d²_c/(2s²) + log p(s²) ]
```

其中 `p(s²)` 由训练 shard 的逐样本残差方差拟合成 200 bins 直方图，离线建成
`g(d²)` 查表。该步骤把本地 BER 从 `0.006884` 降到 `0.006678`。

### 3. 判决：bit-MAP 而不是码字级 ML

评估指标是 **bit BER**，不是码字错误率。码字级 ML 最小化“整词出错率”，而逐 bit
后验可以取得更低的 bit BER：

```text
P(bit_i=1|y) = Σ_{c∈C, c_i=1} exp(logw_c) / Σ_{c∈C} exp(logw_c)
x̂_i = 1  iff  P(bit_i=1|y) ≥ 0.5
```

这是从 `0.00724` 突破到 `0.00692` 的关键。最终输出约 3% 的行不是合法码字，
但官方只按 bit BER 计分，这是 bit 级最优决策的正常结果。

### 4. 神经网络探索与结论

我们尝试过但不采用的方向：

| 方向 | 结果 |
|---|---|
| 端到端 MLP / CNN / Transformer | 无法稳定学到 GF(2) 约束，BER 远差于 OSD |
| Syndrome-NN | 预测不可用 |
| 候选打分 NN / LightGBM 重排 | 候选内 AUC 很高，但 bit-MAP BER 变差 |
| NN 信道状态估计 | 与 scale-mixture 融合后无增益 |
| 全量 64M 数据 + 超大 NN | 数据量增大后 `p(s²)` 已饱和，未看到突破迹象 |

结论：本赛题的瓶颈不是“网络不够大”，而是候选似然已经把可用的区分信息消耗殆尽。
最终仓库只保留统计解码主链路；NN 实验作为研究记录，不进入生产路径。

## 📊 核心挑战与解决方案

1. **64M 样本 / 82 GB 数据 I/O**
   完整训练集 500 个 shard、约 82 GB。复现只需要 shard 000/001；`p(s²)` 在
   12.8 万样本即饱和，增加到 640 万样本 BER 不变。仓库提供 `scripts/download_data.py`，
   数据本身不入库。

2. **候选覆盖与排序**
   order-3 在 1000 样本诊断中 100% 覆盖真码字；剩余错误主要来自排序和后验推断。
   独立复核进一步发现，order-4 仍能补回极少数高权重远候选（10 万行少 51 bit）；
   order-5 因远候选累积偏差反而变差。

3. **码字级 ML 的地板**
   在“逐样本独立、码字级判决”框架下，`0.00724` 是 ML 下界；改用 bit-MAP 后
   突破到 `0.00692`。

4. **混合 SNR 信道**
   逐样本 `s²` 变化是主要非高斯来源；scale-mixture 边缘化是理论正确的似然，
   把 BER 再降约 `0.0002`。

5. **神经网络难以利用**
   NN 可以在候选内找到额外排序信号（AUC 提升），但无法改善逐 bit 的 0.5 阈值判决；
   集成后 BER 严格变差。

## 📁 目录结构

```text
.
├── bit_map_decoder_fast.py        # 早期 bit-MAP 解码器
├── scale_mixture_submit.py        # order-3 + scale-mixture 生产脚本
├── scale_mixture_submit_o45.py    # order-4 + scale-mixture 主推脚本
├── validate_submission.py         # 提交格式 / 校验矩阵检查
├── requirements.txt
├── DATA.md                        # 数据来源与放置方式
├── ATTRIBUTION.md                 # 第三方素材与授权边界
├── docs/                          # 赛题、方法演进、极限分析、独立复核
├── results/                       # 参考提交（gzip）
├── scripts/
│   └── download_data.py           # 训练 shard 下载
├── tests/
│   └── test_validate_submission.py
├── LICENSE                        # Apache-2.0（作者代码）
└── LICENSE-docs                   # CC BY 4.0（文档）
```

> 官方 baseline 资源（`baseline.py`、`baseline.zip`、`channel_calibration.npz`）
> 因官方未附开源许可证，不随仓库分发；获取与放置方式见 [DATA.md](DATA.md)。

## 🚀 快速开始

### 1. 安装依赖

```bash
python -m pip install -r requirements.txt
```

只依赖 `numpy` 和 `pandas`；生产推理不需要 torch/GPU。

### 2. 准备数据

```bash
# 下载复现所需的训练 shard 000/001 到 ./train/
python scripts/download_data.py --shards 0-1

# 把官方 public_test.zip 放到仓库根目录，然后解压到 test/
python -c "import zipfile; zipfile.ZipFile('public_test.zip').extractall('test')"
```

详细说明见 [DATA.md](DATA.md)。`train/`、`test/`、`public_test.zip` 都已在
`.gitignore` 中，不会被提交。

> 两个生产脚本的输入路径不同：order-3 脚本直接读根目录的 `public_test.zip`；
> order-4 脚本读解压后的 `test/competition_data/test_noisy_y_public.csv`。

### 3. 复现主推提交（order-4）

```bash
python scale_mixture_submit_o45.py --mode infer --output submission_o4.csv
python validate_submission.py submission_o4.csv --expect-rows 100000 --skip-parity
```

第二条命令输出 `RESULT: PASS` 才表示格式可提交。`--skip-parity` 是必要的：
bit-MAP 输出按 bit 最优，约 3% 的行不是合法码字，官方评测只按 bit BER 计分。

完整 10 万行 order-4 推理建议 24 核 CPU，约 45 分钟；`--nproc 8` 约 2 小时，
内存占用也更低。

### 4. 复现 order-3（排行榜 0.00675 同款）

```bash
python scale_mixture_submit.py --mode infer --output submission_sm.csv
python validate_submission.py submission_sm.csv --expect-rows 100000 --skip-parity
```

### 5. 只校验参考提交

```bash
gzip -dk results/submission_o4.csv.gz
python validate_submission.py results/submission_o4.csv --expect-rows 100000 --skip-parity
```

Windows 没有 `gzip` 时：

```bash
python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('results/submission_o4.csv.gz','rb'), open('results/submission_o4.csv','wb'))"
```

## 📈 方法演进

| 阶段 | 方法 | BER |
|---|---|---:|
| 官方 baseline | OSD-2 + 信道校准 | 0.01272 |
| OSD 调参 | list_size=32, order=2 | 0.00724 |
| bit-MAP | OSD order-3 候选 + 逐 bit 后验 | 0.00692 |
| scale-mixture（order-3） | 边缘化逐样本 `s²` | **0.00675（排行榜）** |
| scale-mixture（order-4） | 增加全部 4 位翻转候选 | **0.006750（本地 10 万行）** |

## 📚 文档

- [赛题说明](docs/赛题.md)
- [提升过程](docs/提升过程.md)：从 0.01272 到 0.00675 的逐阶段记录
- [极限分析](docs/极限分析.md)：三层“地板”与合法单样本解码边界
- [独立复核结论](docs/独立复核结论.md)：order-4 剩余空间与 oracle σ² 上界
- [路线 A/B 验证报告](docs/路线AB验证报告.md)：全量数据 + NN 与压极限路线
- [数据说明](docs/数据说明.md)

## ⚠️ 已知限制

- 训练数据和公开测试数据不随仓库分发；需要按 [DATA.md](DATA.md) 自行准备。
- order-4 推理内存占用较高，默认 `--nproc 24`；内存不足时用 `--nproc 8` 或更低。
- bit-MAP 输出不保证每行都是合法码字，这是 bit 级最优决策的正常结果。
- 官方 baseline 资源（`baseline.py`、`baseline.zip`、`channel_calibration.npz`）
  与公开测试集来自赛事主办方。官方发布时未附开源许可证，因此本仓库不包含这些
  文件；获取方式见 [DATA.md](DATA.md)，授权边界见 [ATTRIBUTION.md](ATTRIBUTION.md)。
  如果官方有正式许可证，以官方版本为准。
- 排行榜更高名次与本项目复现的合法单样本下限存在差距；具体原因无法从公开信息
  判断，本文档不做针对任何队伍的事实指控。

## 📄 许可

- 作者编写的代码：Apache-2.0，见 [LICENSE](LICENSE)。
- 文档：CC BY 4.0，见 [LICENSE-docs](LICENSE-docs)。
- 官方 baseline、校验矩阵、数据集与测试集的版权归各自权利人所有，且均不随
  本仓库分发；详见 [ATTRIBUTION.md](ATTRIBUTION.md)。
