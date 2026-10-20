# 路线 A（全量数据 + NN）与路线 B（压极限）验证报告

> 日期：2026-08-05　环境：torch2（torch 2.11.0+cu128，numpy 1.26.4，RTX 5060 8GB）
> 目的：按用户要求先做 A（全量数据路线）再做 B（合法压极限），用大样本 + GPU
> 严格检验"64M 数据能否突破 0.0067"这一项目此前唯一未验证的假设。

## 结论速览

| 实验 | 结果 | 结论 |
|---|---|---|
| A1 基线复现（torch2 纯 numpy 流水线） | global-Gaussian 0.006886 / scale-mixture 0.006673 | 与文档 0.006884/0.006678 一致 |
| A2 p(s2) 大样本拟合 | 128k=0.006673；1.1M=0.006677；6.3M=0.006677 | 已饱和，更多数据零增益 |
| A3 NN 学习式似然探针（2M 候选行，GPU） | 候选内 AUC 0.851 到 0.982，但集成进 bit-MAP 后 BER 0.006673 到 0.006717+（严格变差） | 大样本 NN 前提被证伪 |
| B 条件化逐样本 sigma2（1.1M 样本拟合） | 8/16/64 分箱：0.006687 / 0.006706 / 0.006695 | 均差于纯 scale-mixture 0.006673，无增益 |

## 一、A1：复现基线（环境可信性验证）

用 torch2 重写了纯 numpy 解码流水线（`ecc_common.py` + `ecc_validate.py`，
无 pandas，CSV 用 np.loadtxt）。验证口径：shard_000 前 10000 行，
`rng(1234)` 打乱；p(s2) 拟合在独立 shard_001（比文档更严格，文档拟合含验证行）。

- global-Gaussian bit-MAP：0.006886（文档 0.006884）
- scale-mixture bit-MAP：0.006673（文档 0.006678）

## 二、A2：p(s2) 拟合数据量扫描

| 拟合数据 | BER（10k 同口径） |
|---|---|
| shard_001（12.8 万） | 0.006673 |
| shard 2-10（115 万） | 0.006677 |
| shard 2-50（640 万） | 0.006677 |

结论：p(s2) 直方图在 12.8 万样本即饱和；数据量 x50 后 BER 不变（差 2 bit，
噪声级）。"下载全部 88GB 提升 p(s2)" 方向不成立。

## 三、A3：NN 学习式似然探针（本次核心实验）

### 设计
- 数据：shard 2-4 抽 20,000 样本，每样本取 OSD order-3 全部 5489 候选按 d2 排序
  的 top-100，共 200 万 (y, 候选) 行，特征 = [d2, 逐坐标残差2(64), y_i*s_i(64),
  |y_i|(64)]，标签 = 候选是否为真实码字。
- 模型：MLP(193-256-128-1)，BCE + 正类加权，GPU 3 epochs（约 10 秒/轮）。

### 结果
| 指标 | -d2 基线 | NN |
|---|---|---|
| 候选内 AUC（top-100） | 0.85142 | 0.98164（+0.13） |
| top-1 命中率 | 0.8535 | 0.8472（反而更低） |

NN 确实在 d2 之外找到了可学的排序信号（AUC 大增），但集成进 bit-MAP 后验：

| alpha（logw = g(d2) + alpha*NN 分数） | BER |
|---|---|
| 0（纯 scale-mixture） | 0.006673 |
| 0.1 | 0.006717 |
| 0.3 | 0.006811 |
| 1.0 | 0.007078 |
| 3.0 | 0.007577 |

任何 alpha>0 都严格变差。与文档 LightGBM 重排结论完全一致（AUC 高但 BER 变差），
且本次是在 10 倍数据规模 + GPU 下复现。根因：候选似然已经由 scale-mixture
Gaussian（即 g(d2)）精确刻画，NN 的"额外信号"只是把后验权重扭曲掉，并不能
改善逐 bit 最优的 0.5 阈值判决。

**结论：A 路线（64M 数据 + 大 NN 学非高斯信道）的前提被实验证伪。** 即便
64M 全量下载完，NN 也只能在候选内找到已被 g(d2) 消耗殆尽的信号。

## 四、B：条件化逐样本 sigma2

用训练集拟合 p(s2 | |y|2)（|y|2 是测试集自身可用的统计量，合法），
按 |y|2 分箱替换全局 p(s2) 做 scale-mixture 边缘化：

| 分箱数 | BER |
|---|---|
| 8 | 0.006687 |
| 16 | 0.006706 |
| 64 | 0.006695 |
| 纯 scale-mixture（对照） | 0.006673 |

全部差于对照（且差异 <1sigma，噪声级）。边缘化本身已正确处理逐样本 sigma2 的
不确定性；用噪声很大的 |y|2 去条件化只会引入更多误差。文档中 tau=0.8 的
0.006673 已是最优。

## 五、最终结论

1. 0.00675（榜）/ 0.006673（本地同口径）就是合法框架的极限。A、B 两条
   此前"未验证"的方向在大样本 + GPU 下均被证伪，且与文档理论分析一致。
2. 不建议再提交：本轮没有任何变好的方案；当前最优仍是 scale-mixture bit-MAP
   （参考提交见 `results/`）。
3. 想低于 0.0067，合法路径已穷尽；前三名（<=0.00186）与本项目复现下限存在明显差距，
   公开信息不足以解释，可能涉及更大规模训练/算力、评测集差异或测试集信息使用，需向主办方求证。

## 六、可复现命令（torch2，历史研究工作区记录）

> 以下命令对应原始研究工作区，部分脚本未随本仓库分发；主复现请使用
> README 的 order-4 命令。

```powershell
cd iflytek-ecc-decode
# 1) 生成 10k 验证候选缓存（35 秒）
python ecc_validate.py cache --val-shards 0 --val-rows 10000 --seed 1234 --cache val10k.npz
# 2) 基线：高斯 / scale-mixture / 条件 sigma2
python ecc_validate.py score --cache val10k.npz --fit-shards 1 --like all
python ecc_validate.py score --cache val10k.npz --fit-shards 2-10 --fit-dir <research-data>/train --like cond --n-bins 8
# 3) NN 探针
python ecc_rerank_data.py --shards 2-4 --base-dir <research-data>/train --n-samples 20000 --K 100 --out rerank_train20k.npz
python ecc_rerank_train.py --data rerank_train20k.npz --epochs 3 --out nn_rerank.pt
python ecc_rerank_integrate.py --cache val10k.npz --model nn_rerank.pt --K 200 --alpha 0,0.1,0.3,1,3
```

新增文件：`ecc_common.py`、`ecc_validate.py`、`ecc_rerank_data.py`、
`ecc_rerank_train.py`、`ecc_rerank_integrate.py`（均在 原始研究工作区根目录）。
