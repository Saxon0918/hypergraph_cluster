# Hypergraph Cluster Expansion

非均匀超图谱热力学的可复现研究代码。实现严格采用实施说明统一后的模型：

\[
G=H^\top H,\qquad U(\beta)=+\partial_\beta\log Z,\qquad
S(\beta)=\log Z-\beta U,\qquad F(\beta)=-\log Z/\beta.
\]

核心 `G` 保留超边大小对角线，不添加单位阵、不清空对角、不默认做拉普拉斯归一化。顶点侧 `Q=HHᵀ` 只作为同一完整谱的精确加速，并恢复原始 `M` 个谱状态中的全部零模。

## 当前实现

- 数学内核：二值稀疏关联矩阵、G/Q 自动选边、完整特征值、稳定 `logZ/U/S/F`、谱方差与热容。
- 闭合游走：`W0..W3`、`A2/O2/A3/B3/T3` 有序和分解；大 `M` 时不强制构造 `M×M` 的 G。
- 近似与反演：`κ1..κ3`、`U1/U2`、`S2/S3`、有理化局部逆解、带状态的精确二分参照。
- 数据：HIC 单标签文本、ScHoLP timestamped simplices、SocioPatterns 接触表、逐 20 秒图的全部极大团、确定性清洗和审计。
- 零模型：保持大小的 N1；按固定尝试步数推进、同时保持顶点度和超边大小的二值 2×2 交换链 N2。
- 分类：BasicStats、累积量、低阶分解、SizeOnlyThermo、GramThermo、组合特征、edge/vertex NetLSD；同构分组和无泄漏嵌套交叉验证。
- E2 性能：NetLSD 与可复用谱曲线按图缓存；每个内层折的特征和标准化只计算一次并复用于全部超参数；使用 L2 `LinearSVC`（`squared_hinge`、`dual=auto`）；每完成一个数据集/方法即原子保存检查点，可中断后续跑。
- E1 性能：谱与三阶矩复用同一 Gram 算子；N1 批次和四条 N2 链支持进程并行；交换链使用批量等分布提议和常数时间指纹查重；每批/每链保存检查点，并从检查点流式合并正式长表。
- E3 性能：全部图保留前向/逆向误差分析，计时使用按算子维度确定性覆盖的子样本；一次预处理复用于 `J=1/12/80`；超出计算维度上限的 Exact-G 显式记为 `not_run`；逐图检查点支持续跑。
- 外部基线：HIC B8/B9、MoCHy、RethinkHGK 通过冻结 commit 的隔离环境 JSON 协议接入；未配置作者代码时输出 `not_run`，绝不填入论文公开分数。
- 实验入口：数据审计、E1 零模型、E2 分类、E3 近似/反演/效率、E4 学校动态、从保存结果独立作图。
- 输出：CSV/Parquet 长表、来源/划分/环境/完整配置、失败 JSONL，以及 PDF+PNG 图。

代码不会伪造正式实验结果。仓库中的 `tests/fixtures` 只用于 smoke test，不能进入论文结果。

## 解释器与运行方式

本机验证使用用户指定解释器：

```bash
/opt/anaconda3/envs/pytorch/bin/python --version
# Python 3.9.19
```

该环境已包含全部运行依赖。无需安装即可从仓库根目录运行：

```bash
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="/tmp/hce-matplotlib"

/opt/anaconda3/envs/pytorch/bin/python -m hce.cli self-check
/opt/anaconda3/envs/pytorch/bin/python scripts/verify_runtime.py
```

如需安装命令 `hce`，可在已有依赖的前提下执行：

```bash
/opt/anaconda3/envs/pytorch/bin/python -m pip install --no-deps --no-build-isolation -e .
```

`requirements-pytorch.lock` 记录本次实际验证的运行时版本。`pytest` 是开发可选依赖；指定环境当前未安装它，因此本次使用 `self-check`、`verify_runtime.py` 和端到端 smoke 命令验证。安装开发依赖后可再运行 `python -m pytest -q`。

## 可移植数据与服务器运行

正式输入统一放在 `data/raw/`，实验输出统一写入 `runs/<run_id>/`；两者均可由 Git 同步。`configs/local_*.yaml` 只使用项目根目录相对路径，代码会根据配置文件位置寻找 `pyproject.toml` 来确定项目根，因此服务器上的绝对目录可以不同，甚至可以从项目外调用命令。

当前目录约定：

- `data/raw/hic/`：HIC 的三个实验输入文本。
- `data/raw/email-Enron-full/`、`data/raw/congress-bills-full/`：ScHoLP 完整版本。
- `data/raw/primaryschool/`、`data/raw/contact-primary-school/`：学校时序数据。
- `runs/`：配置快照、环境、表格和图片等实验产物。

下载压缩包 `data/raw/downloads/` 和可重建缓存 `data/processed/` 不进入 Git。审计命令会把每个实际输入的 SHA256 写入运行目录。向公开仓库推送数据前，必须另行核实各数据源的再分发许可。

服务器运行示例：

```bash
git clone <your-repository-url> hypergraph_cluster
cd hypergraph_cluster
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="/tmp/hce-matplotlib"

python -m hce.cli self-check
python -m hce.cli audit --config configs/local_e1.yaml
python -m hce.cli classify --config configs/local_e2.yaml
python -m hce.cli nulls --config configs/local_e1.yaml
python -m hce.cli approximation --config configs/local_e3.yaml
python -m hce.cli temporal --config configs/local_e4.yaml
```

完成后在服务器提交 `runs/`，本机 `git pull` 即可取得机器可读结果和图。若将来单个结果文件超过托管平台限制（GitHub 为 100 MB），应为结果文件启用 Git LFS，而不是删减实验数据。

E1–E3 都会持续输出进度并写入 `checkpoints/`。使用完全相同的配置重新执行命令时，已完成的 E1 样本/E2 方法/E3 样本会直接恢复；E1 的 N1 批次和 N2 单链也可从部分检查点继续。E1/E3 检查点属于服务器本地的可重建中间状态，默认不进入 Git；最终 CSV/Parquet 结果仍正常同步。`classification_runtime.csv` 和 E3 的 `runtime.csv` 保存分阶段实际耗时。

直接 URL 可用清单下载器；Google Drive 若返回确认 HTML 会被拒绝：

```bash
/opt/anaconda3/envs/pytorch/bin/python -m hce.cli fetch \
  --manifest configs/official_sources_manifest.json \
  --output-dir data/raw/downloads
```

HIC 官方相对路径是：

- `data/hypergraph/IMDB/IMDB_dir_genre.txt`
- `data/hypergraph/STEAM/stream_player.txt`（文件名确为 `stream_player`）
- `data/hypergraph/IMDB/IMDB_dir_form.txt`

## 实验命令

使用已配置好的项目相对路径运行：

```bash
/opt/anaconda3/envs/pytorch/bin/python -m hce.cli audit --config configs/local_e1.yaml
/opt/anaconda3/envs/pytorch/bin/python -m hce.cli nulls --config configs/local_e1.yaml
/opt/anaconda3/envs/pytorch/bin/python -m hce.cli classify --config configs/local_e2.yaml
/opt/anaconda3/envs/pytorch/bin/python -m hce.cli approximation --config configs/local_e3.yaml
/opt/anaconda3/envs/pytorch/bin/python -m hce.cli temporal --config configs/local_e4.yaml
/opt/anaconda3/envs/pytorch/bin/python -m hce.cli report --run-dir runs/<run_id>
```

在提交正式长任务前，可先运行不进入论文结果的 pilot：

```bash
/opt/anaconda3/envs/pytorch/bin/python -m hce.cli nulls --config configs/local_e1_quick.yaml
/opt/anaconda3/envs/pytorch/bin/python -m hce.cli approximation --config configs/local_e3_quick.yaml
```

端到端 smoke test 使用 12 个合成小图，覆盖全部本地流水线：

```bash
/opt/anaconda3/envs/pytorch/bin/python -m hce.cli audit --config configs/smoke.yaml
/opt/anaconda3/envs/pytorch/bin/python -m hce.cli classify --config configs/smoke.yaml
/opt/anaconda3/envs/pytorch/bin/python -m hce.cli nulls --config configs/smoke.yaml
/opt/anaconda3/envs/pytorch/bin/python -m hce.cli approximation --config configs/smoke.yaml
/opt/anaconda3/envs/pytorch/bin/python -m hce.cli temporal --config configs/smoke.yaml
```

## 关键实验约束

- 分类的 `c_D` 在每个 inner-train 内单独拟合；标准化也只看该训练部分。外层测试只在最终重拟合后评估一次。
- 纯结构同构样本用带两类节点颜色的关联二部图做候选哈希，再以精确同构确认并分组。
- N2 每步抽取两个现有关联，以尝试步数推进；拒绝也算链的一步。每个保存状态重新检查大小、度、二值和简单性。
- E1 的正式尺度使用大小统计不变量；主统计是冻结 `[0.01,10]` 区间上的归一化熵对数尺度积分，不逐点挑峰。
- E3 的 `x=βλmax` 仅用于误差诊断。效率路线分别计入算子构造、Exact-G/Exact-small-side 谱分解或 Moment-3 矩计算、以及多尺度查询；不会用完整谱先算累积量。运行时间子样本按每个数据集的算子维度等距覆盖选择，完整清单写入 `benchmark_manifest.csv`。
- 学校数据必须先在每个原始 20 秒接触图找极大团，再汇总到 10 分钟窗口。空窗口是缺失状态，不填 `S=0`。

## 结果与失败语义

默认输出到 `runs/<run_id>/`。正式表包括 `preprocess_audit.csv`、`sample_statistics.parquet`、`moments.parquet`、`curves.parquet`、`null_diagnostics.csv`、`null_statistics.csv`、`cv_predictions.parquet`、`cv_metrics.csv`、`approximation_errors.parquet`、`runtime.csv`、`temporal_windows.parquet`、`resolved_config.yaml` 和 `failures.jsonl`。

`M=0`、显著负特征值、内存预算不足、时间原点未核实、外部作者基线缺失、反演无效和随机链诊断不足都有显式状态；这些行不会被静默删除，也不会用 0 或作者论文成绩填充。

## 研究解释边界

这套描述符对同谱超图不可区分；增加孤立顶点不改变 G。`T3` 是三条不同超边两两重叠乘积的有序总和，不是三重交集。有限维曲线的集中不自动构成相变；较高熵只表示当前尺度下谱概率更分散。源码附带真实输入不等于 E1–E4 已得到科学结论；只有实际运行、审计和统计输出才构成实验结果。
