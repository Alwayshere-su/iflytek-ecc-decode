# 参考提交

这里保存两份完整 10 万行提交的 gzip 压缩版，方便读者不重新跑推理也能检查
输出格式和逐位结果。

| 文件 | 方法 | 说明 |
|---|---|---|
| `submission_o4.csv.gz` | order-4 + scale-mixture bit-MAP | 当前主推；10 万行本地验证 BER 0.006750 |
| `submission_sm_t1.00.csv.gz` | order-3 + scale-mixture bit-MAP，T=1.0 | 对应排行榜 BER 0.00675 的方法 |

## 解压与校验

```bash
gzip -dk results/submission_o4.csv.gz
python validate_submission.py results/submission_o4.csv --expect-rows 100000 --skip-parity
```

Windows 没有 `gzip` 时：

```bash
python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('results/submission_o4.csv.gz','rb'), open('results/submission_o4.csv','wb'))"
```

`--skip-parity` 是必要的：bit-MAP 输出按 bit 最优，约 3% 的行不是合法码字，
官方评测只按 bit BER 计分。

## 注意

- 两份提交都对应同一批 10 万行公开测试输入（`id=0..99999`）。
- `submission_o4.csv.gz` 与 order-3 版本相比有约 790 个 bit 不同；order-4 是
  候选扩展后的主推版本。
- 解压后的 CSV 已被 `.gitignore` 排除，不会误提交。
