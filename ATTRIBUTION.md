# 第三方素材与授权边界

本仓库包含作者在 2026 讯飞纠错码解码挑战赛中的代码、实验记录和文档。
赛事官方/第三方素材的版权归各自权利人所有，本仓库不重新分发它们。

## 赛事官方素材（不随仓库分发）

- `baseline.py`：官方提供的 OSD-2 baseline 参考实现。
- `baseline.zip`：官方 `POLAR_N64_K32` 校验矩阵。
- `channel_calibration.npz`：由官方 baseline 流程在训练数据上生成的校准缓存。
- 公开测试集、赛题说明：2026 讯飞纠错码解码挑战赛主办方提供。

由于官方发布上述素材时没有附带开源许可证，本仓库**不包含**这些文件，也不在
Git 历史中保留它们。复现时请从官方渠道获取，并按 [DATA.md](DATA.md) 放在仓库
根目录（`.gitignore` 已排除，不会误提交）。如果官方发布正式许可证或使用条款，
以官方版本为准。公开测试集和训练数据同样**不随本仓库分发**。

## 训练数据

训练数据来自 HuggingFace 数据集
[`aprofeta/ecc-dataset`](https://huggingface.co/datasets/aprofeta/ecc-dataset)。
数据集版权和使用条款归数据集作者/发布方所有。本仓库只提供下载脚本，不包含
任何 shard 文件。

## 作者贡献

以下内容为作者在本赛事中的实现与分析：

- `scale_mixture_submit.py`、`scale_mixture_submit_o45.py`：scale-mixture
  bit-MAP 列表解码器；
- `bit_map_decoder_fast.py`：早期 bit-MAP 解码器；
- `validate_submission.py`：提交格式与校验矩阵检查；
- `scripts/download_data.py`：数据下载脚本；
- `docs/`：方法演进、极限分析、独立复核与路线 A/B 报告；
- `results/`：参考提交的压缩版本。

作者编写的代码采用 Apache-2.0，文档采用 CC BY 4.0；官方素材不自动适用这两个
许可证，仍归各自权利人所有。

## 关于排行榜的说明

文档中讨论的“合法单样本解码下限”是作者基于公开数据、公开规则和本地复现实验
得到的结论。排行榜更高名次与本项目下限存在差距；具体原因无法从公开信息判断，
可能涉及更大规模训练/算力、评测集差异或测试集信息使用。本文档不做针对任何
队伍的事实指控，也不声称掌握未公开的评测细节。
