# 数据准备

本仓库不包含训练集和公开测试集，也不 redistributing 官方数据。下面的路径与
`scale_mixture_submit*.py` 的默认相对路径一致。

## 1. 训练数据

复现所需的两个 shard 来自 HuggingFace 数据集
[`aprofeta/ecc-dataset`](https://huggingface.co/datasets/aprofeta/ecc-dataset)：

```bash
python scripts/download_data.py --shards 0-1
```

脚本默认下载到 `./train/`，得到四个文件：

```text
train/train_codeword_x_shard_000.csv
train/train_codeword_x_shard_001.csv
train/train_noisy_y_shard_000.csv
train/train_noisy_y_shard_001.csv
```

每个 shard 是 128,000 对样本：`train_codeword_x_shard_XXX.csv` 是正确答案，
`train_noisy_y_shard_XXX.csv` 是含噪信号。更多 shard 可以用
`--shards 0-9` 或 `--shards 0-499` 下载。

脚本默认使用 `hf-mirror.com` 镜像；如果网络环境不同，可以用 `--base-url`
指向其他镜像或本地服务。

## 2. 公开测试集

把赛事官方提供的 `public_test.zip` 放在仓库根目录，然后解压：

```bash
python -c "import zipfile; zipfile.ZipFile('public_test.zip').extractall('test')"
```

应得到：

```text
test/competition_data/test_noisy_y_public.csv
```

如果你拿到的是未压缩的 CSV，直接放到同一路径即可。该文件只包含 `id` 和
`y_0..y_63`，不包含真实 `bit_*` 标签。

- `scale_mixture_submit_o45.py` 读取 `test/competition_data/test_noisy_y_public.csv`
- `scale_mixture_submit.py` 和 `bit_map_decoder_fast.py` 读取根目录的 `public_test.zip`

## 3. 数据是否会进 Git

不会。`.gitignore` 已排除：

```text
train/
test/
public_test.zip
hf_data/
baseline.py
baseline.zip
channel_calibration.npz
*.part
```

训练数据约 338 MB（shard 000/001），公开测试 CSV 约 119 MB；两者都超过或接近
GitHub 单文件 100 MB 的限制，因此只保留下载脚本和目录说明。

## 4. 官方 baseline 资源（不随仓库分发）

官方发布的 baseline 包没有附带开源许可证，因此本仓库不包含、也不在 Git 历史中
保留以下文件：

- `baseline.py`：官方 OSD-2 baseline 参考实现；
- `baseline.zip`：官方 `POLAR_N64_K32` 校验矩阵；
- `channel_calibration.npz`：由 baseline 在训练数据上生成的校准缓存。

从赛事官方渠道取得 `baseline.py` 与 `baseline.zip` 后放在仓库根目录。首次运行
`python baseline.py --max-rows 100 --output submission_debug.csv` 会生成
`channel_calibration.npz`，之后可用 `--reuse-calibration` 复用。这些文件已加入
`.gitignore`，不会被误提交。

缺少 `baseline.zip` 时，`validate_submission.py` 会跳过校验矩阵检查并给出警告；
`bit_map_decoder_fast.py` 和 `scale_mixture_submit*.py` 需要该文件才能运行。

## 5. 数据来源与授权

- 训练数据：`aprofeta/ecc-dataset`，版权与使用条款归数据集作者/发布方所有。
- 公开测试集、官方 baseline、POLAR_N64_K32 校验矩阵：2026 讯飞纠错码解码挑战赛
  主办方提供，版权归主办方所有。
- 本仓库只分发作者编写的代码、文档和压缩后的参考提交；不重新分发上述数据与
  官方 baseline 资源。如官方发布正式许可证，以官方条款为准。详见
  [ATTRIBUTION.md](ATTRIBUTION.md)。
