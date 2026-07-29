# RMBench Step Visualization 批量适配计划

## 一、目标与数据约定

- 新增与 `scripts/language_table` 风格一致的 RMBench 专用批量可视化工作流，支持多机多卡、确定性分片、断点续跑、失败隔离、日志记录和 dry-run。
- 默认数据根目录为 `/mnt/umm/users/zuojing/code/RMBench/data/data`，同时允许通过 `--data-dir` 覆盖。
- 按实际数据结构发现样本：`{task}/demo_clean/video/episodeN.mp4` 和 `instructions/episodeN.json`。
- 每个 episode 使用头部相机视频的第一帧作为输入图像；RMBench 的数据生成代码已确认 `video/episodeN.mp4` 来自 `head_camera/rgb`，因此无需额外读取约 90 MB/episode 的 HDF5。
- 默认读取 `instructions/episodeN.json` 中的 `seen[0]` 作为 prompt，并支持 `--instruction-split unseen`。当前 600 个 episode 的 seen/unseen 均各有一条且内容相同。
- 不修改 RMBench 仓库及其数据，只读取源数据并在当前仓库的输出目录生成标准化输入与可视化结果。

## 二、实现改动

### 1. 公共批处理能力

新增内部批处理公共模块，将 Language-Table 中可复用的能力迁入其中，并保持原有 Language-Table 行为不变：

- episode 范围解析，如 `all`、`0:10`、`0:20:2,49`。
- GPU 自动检测和显式 GPU 列表校验。
- 基于全局样本顺序、`node-rank`、`num-nodes` 和本机 GPU 的确定性无重复分片。
- 原子写入 `run.json`、assignment 文件和结果 metadata。
- 运行清单一致性校验、`--overwrite` 行为和 dry-run 只读保证。
- worker 生命周期、独立日志和退出码汇总逻辑仍由各数据集入口负责，避免把数据集解析耦合进公共模块。

### 2. RMBench 专用工作流

新增 `scripts/rmbench/visualize.py`，提供数据路径、输出路径、任务和 episode 筛选、seen/unseen 指令选择、模型与生成参数、多机多卡参数、`--extract-only`、`--overwrite` 和 `--dry-run`。

- 自动发现所有任务和数值化排序的 `episodeN`，严格校验视频、instruction 文件以及所选 instruction 字段。
- `--tasks` 默认 `all`；`--episodes` 对每个所选任务应用同一个范围，默认 `all`。
- 样本按“任务名、episode 数字”稳定排序，内部 ID 使用 `task_name__episode_000000`，避免不同任务间 episode 编号冲突。
- dry-run 输出总样本数、每个任务数量及每张 GPU 的分配预览，不创建输出目录。
- 抽取阶段仅读取 MP4 第一帧，转换为 RGB PNG；校验图像尺寸有效且满足模型的 16 像素对齐要求。
- 推理阶段每个 GPU 启动一个独立 worker，每个 worker 只加载一次模型，然后依次处理分配到的样本。
- 单个 episode 失败时记录错误并继续处理同一 worker 的其他样本；仅在完整成功后原子替换最终目录。
- 已完成结果的来源、prompt、instruction split 和生成参数全部一致时跳过；不一致时拒绝混写，除非显式使用 `--overwrite`。

### 3. 启动、输出与文档

- 新增 `scripts/rmbench/launch.sh`，从 `DATA_DIR`、`OUTPUT_ROOT`、节点、GPU 和模型环境变量构造多机命令。
- 输出统一使用 `run.json`、`inputs/{task}__episode_{id}/`、`episodes/{task}__episode_{id}/` 和 `runtime/{assignments,logs}/`。
- 新增 `docs/workflows/rmbench.md`，并在 README 增加 RMBench 入口。

## 三、接口与兼容性

- 新增 RMBench CLI，不改变现有模型管线公开调用方式。
- Language-Table 仅调整公共辅助函数的导入位置，现有 CLI 参数、输出格式和运行语义保持兼容。
- RMBench 的运行和 episode metadata 记录数据来源、任务、episode、instruction split、prompt 和完整生成参数。
- 不新增 `h5py` 或 OpenCV 依赖；第一帧读取复用项目已有的 `imageio[ffmpeg]` 与 Pillow。

## 四、测试与验收

- 单元测试覆盖任务发现、episode 排序和筛选、样本 ID、prompt 解析、错误输入、多机多卡分片、恢复和 dry-run。
- 使用小型临时视频验证首帧提取；缺少 ffmpeg 后端时明确记录跳过原因。
- 回归现有 Language-Table 和 custom workflow 测试。
- 执行 `python -m unittest`、`python -m compileall`、`ruff check`、`bash -n` 和 `git diff --check`。
- 模型和 GPU 可用时执行单 episode 冒烟测试；否则完成 dry-run、extract-only 和命令构造验收并记录限制。

## 五、执行顺序与默认约定

1. 落盘本计划。
2. 提取通用批处理辅助逻辑并回归 Language-Table。
3. 实现 RMBench 数据解析、抽取和多机多卡推理。
4. 添加启动脚本、测试和文档。
5. 完成静态检查、dry-run、extract-only 和条件式 GPU 冒烟测试。

默认采用已确认的选择：使用专用 RMBench 批量工作流、头部相机首帧、`seen` 指令和多机多卡支持；不修改 RMBench 数据、不下载模型、不创建 commit。
