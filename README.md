# 讯飞纠错码解码挑战赛 · POLAR(64,32)

[![CI](https://github.com/Alwayshere-su/iflytek-ecc-decode/actions/workflows/ci.yml/badge.svg)](https://github.com/Alwayshere-su/iflytek-ecc-decode/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Docs: CC BY 4.0](https://img.shields.io/badge/docs-CC%20BY%204.0-lightgrey.svg)](LICENSE-docs)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)

> 从含噪的 64 维接收信号 `y_0..y_63` 恢复 POLAR(64,32) 码字 `bit_0..bit_63`。
> 排行榜 BER **0.00675（第 5 名）**；本仓库提供可复现的 **scale-mixture bit-MAP 列表解码器**。

## 这是什么

赛题给出经过 AWGN 信道的 64 维接收信号，要求恢复 64 个原始比特。码字满足一个固定的
32×64 校验矩阵。评估指标是 **BER（比特错误率）**，越低越好。

本仓库把解码拆成三步：

1. **OSD 候选生成**：按 `|y|` 可靠性排序，枚举最不可靠比特的 order-3 / order-4 翻转组合；
2. **scale-mixture 似然**：信道是逐样本混合 SNR，对每个样本的噪声功率 `s²` 做边缘化；
3. **bit-MAP 判决**：对每个 bit 计算后验 `P(bit_i=1|y)`，取 0.5 阈值。

关键点是：评估指标是 **bit BER**，不是码字错误率。码字级 ML 最小化“整词出错率”，
而逐 bit 后验可以取得更低的 bit BER——这是从 `0.00724` 突破到 `0.00692` 的原因。

## 结果

| 阶段 | 方法 | BER |
|---|---|---:|
| 官方 baseline | OSD-2 + 信道校准 | 0.01272 |
| OSD 调参 | list_size=32, order=2 | 0.00724 |
| bit-MAP | OSD order-3 候选 + 逐 bit 后验 | 0.00692 |
| scale-mixture（order-3） | 边缘化逐样本 `s²` | **0.00675（排行榜第 5）** |
| scale-mixture（order-4） | 增加全部 4 位翻转候选 | **0.006750（10 万行本地同口径）** |

`results/` 中放了两份参考提交的压缩版：

- `submission_o4.csv.gz`：order-4，当前主推，10 万行本地验证 BER 0.006750；
- `submission_sm_t1.00.csv.gz`：order-3，T=1.0，对应排行榜 0.00675 的方法。

## 快速开始

### 1. 环境

```bash
python -m pip install -r requirements.txt
```

只依赖 `numpy` 和 `pandas`；生产推理不需要 torch/GPU。完整 10 万行 order-4 推理建议
24 核 CPU，约 45 分钟；`--nproc 8` 约 2 小时，内存占用也更低。

### 2. 准备数据

本仓库**不包含训练集和测试集**，数据需要从官方渠道获取。详见 [DATA.md](DATA.md)。

```bash
# 下载复现所需的训练 shard 000/001 到 ./train/
python scripts/download_data.py --shards 0-1

# 把官方 public_test.zip 放到仓库根目录，然后解压到 test/
python -c "import zipfile; zipfile.ZipFile('public_test.zip').extractall('test')"
```

解压后应得到：

```text
train/train_codeword_x_shard_000.csv
train/train_codeword_x_shard_001.csv
train/train_noisy_y_shard_000.csv
train/train_noisy_y_shard_001.csv
test/competition_data/test_noisy_y_public.csv
```

`train/`、`test/`、`public_test.zip` 都已在 `.gitignore` 中，不会被提交。

### 3. 复现主推提交（order-4）

```bash
python scale_mixture_submit_o45.py --mode infer --output submission_o4.csv
python validate_submission.py submission_o4.csv --expect-rows 100000 --skip-parity
```

第二条命令输出 `RESULT: PASS` 才表示格式可提交。

`--skip-parity` 是必要的：bit-MAP 输出按 bit 最优，约 3% 的行不是合法码字，
官方评测只按 bit BER 计分，不要求码字级合法。

### 4. 复现 order-3（排行榜 0.00675 同款）

```bash
python scale_mixture_submit.py --mode infer --output submission_sm.csv
python validate_submission.py submission_sm.csv --expect-rows 100000 --skip-parity
```

order-3 脚本读取根目录的 `public_test.zip`；order-4 脚本读取
`test/competition_data/test_noisy_y_public.csv`。

### 5. 只校验参考提交

```bash
# Git Bash / Linux / macOS
gzip -dk results/submission_o4.csv.gz
python validate_submission.py results/submission_o4.csv --expect-rows 100000 --skip-parity
```

Windows 没有 `gzip` 时可以用 Python：

```bash
python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('results/submission_o4.csv.gz','rb'), open('results/submission_o4.csv','wb'))"
```

## 仓库结构

```text
.
├── baseline.py                    # 官方 OSD baseline（参考）
├── bit_map_decoder_fast.py        # 早期 bit-MAP 解码器（本地验证）
├── scale_mixture_submit.py        # order-3 + scale-mixture 生产脚本
├── scale_mixture_submit_o45.py    # order-4 + scale-mixture 主推脚本
├── validate_submission.py         # 提交格式 / 校验矩阵检查
├── baseline.zip                   # 官方 POLAR_N64_K32 校验矩阵
├── channel_calibration.npz        # baseline 的信道校准缓存
├── requirements.txt
├── docs/                          # 赛题、方法演进、极限分析、独立复核
├── results/                       # 参考提交（gzip）
├── scripts/
│   └── download_data.py           # 训练 shard 下载
├── tests/
│   └── test_validate_submission.py
├── DATA.md                        # 数据来源与放置方式
├── ATTRIBUTION.md                 # 第三方素材与授权边界
├── LICENSE                        # Apache-2.0（作者代码）
└── LICENSE-docs                   # CC BY 4.0（文档）
```

## 已知限制

- 训练数据和公开测试数据不随仓库分发；需要按 [DATA.md](DATA.md) 自行准备。
- 完整 order-4 推理需要较大内存：24 进程约 30–40 GB；内存不足时用 `--nproc 8`。
- bit-MAP 输出不保证每行都是合法码字，这是 bit 级最优决策的正常结果。
- `baseline.py`、`baseline.zip` 和公开测试集来自赛事官方/主办方；授权边界见
  [ATTRIBUTION.md](ATTRIBUTION.md)。如果官方有正式许可证，以官方版本为准。
- 排行榜上更高的名次低于本项目实测的合法单样本解码下限。具体原因无法从公开信息
  判断，可能涉及更大规模训练/算力、评测集差异或测试集信息使用；本文档不做针对
  任何队伍的事实指控。

## 文档

- [赛题说明](docs/赛题.md)
- [提升过程](docs/提升过程.md)：从 0.01272 到 0.00675 的逐阶段记录
- [极限分析](docs/极限分析.md)：为什么 0.00675 已接近合法框架的极限
- [独立复核结论](docs/独立复核结论.md)：order-4 剩余空间与边界复核
- [路线 A/B 验证报告](docs/路线AB验证报告.md)：全量数据 + NN 与压极限路线
- [数据说明](docs/数据说明.md)

## 许可

- 作者编写的代码：Apache-2.0，见 [LICENSE](LICENSE)。
- 文档：CC BY 4.0，见 [LICENSE-docs](LICENSE-docs)。
- 官方 baseline、校验矩阵、数据集与测试集的版权归各自权利人所有；详见
  [ATTRIBUTION.md](ATTRIBUTION.md)。
