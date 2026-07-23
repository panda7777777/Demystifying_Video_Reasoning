# 未提交代码整理与重构计划

## 一、目标与原则

- 整理当前全部未提交改动，重点提升专业性、简洁性、可读性和可维护性。
- 允许彻底调整尚未提交的脚本路径与参数，不保留旧入口兼容脚本。
- 保持核心能力：自定义输入推理、Language-Table 下载与批量推理、多机多卡分片、断点续跑、逐步视频导出和结果后处理。
- 不改动与本次未提交功能无关的既有模型实现；对已修改的核心文件仅做必要的局部整理。
- 本计划写入仓库并获得确认后，才开始修改代码。

## 二、目录、文件与输出结构

### 脚本结构

将当前平铺且职责混杂的脚本整理为：

```text
scripts/
├── custom/
│   └── visualize.py
└── language_table/
    ├── download.py
    ├── visualize.py
    ├── results.py
    └── launch.sh
```

- `custom/visualize.py`：管理文件夹式自定义样例，支持列举、单项和批量推理。
- `language_table/download.py`：下载及校验数据集，不包含个人绝对路径或重复的数据集名称目录。
- `language_table/visualize.py`：负责数据解析、抽取、多机多卡分片、工作进程调度和恢复。
- `language_table/results.py`：通过子命令统一替代零散的结果查询、抽样和帧渲染脚本。
- `language_table/launch.sh`：改为仓库相对路径的通用模板；必填路径通过环境变量或参数传入。
- 删除原有未提交旧入口，不增加转发壳。

### 示例数据

将两组样例规范化放入 `examples/custom_dataset/`：

```text
examples/custom_dataset/
├── README.md
├── maze/
│   ├── input.png
│   └── prompt.txt
└── language_table_memory/
    ├── input.png
    └── prompt.txt
```

- 删除空文件 `chat.txt`。
- 删除根目录重复的 `maze.png`。
- 删除与 `image.png` 内容相同的 `first_frame.png`。
- 统一样例文件名、ID 和说明文档，不保留同一图片的多份副本。

### 统一输出结构

自定义输入与 Language-Table 使用一致的结果约定：

```text
{output_root}/
├── run.json
├── inputs/{sample_id}/
│   ├── input.png
│   └── prompt.txt
├── episodes/{sample_id}/
│   ├── generated.mp4
│   ├── steps/step_000.mp4
│   └── metadata.json
├── analysis/{sample_id}/
│   ├── frames/...
│   ├── montage.png
│   └── metadata.json
└── runtime/
    ├── assignments/
    └── logs/
```

- `run.json` 记录模型、随机种子、分片方式和生成参数。
- `metadata.json` 仅在单个样例成功完成后原子写入，替代含义不透明的 `.done` 文件。
- 恢复运行时校验已有 `run.json`；关键生成参数不一致时拒绝混写。
- 输入、结果、分析产物和运行日志彼此分离。

## 三、代码与接口重构

- 抽取模型构建、默认负向提示词、LoRA 加载、步数解析、回调创建和生成配置为内部公共模块。
- 使用 `GenerationConfig` 统一单样例、批处理和运行清单中的生成参数。
- VBVR 模型路径不再默认为个人目录；选择该模型时要求显式参数或 `VBVR_MODEL_PATH`。
- 文件路径统一使用 `pathlib.Path`，参数统一为标准连字符形式。
- 将 Wan 管线新增参数命名为 `max_denoising_steps`，并校验其范围。
- 每一步至多计算一次 `x0_hat`，由可视化回调与提前停止共享。
- 将数据解析、索引解析和分片拆成可测试的纯函数。
- 统一结果查询、随机抽样和帧渲染子命令。
- 修复 `pyproject.toml` 与 `diffsynth/version.py` 的版本冲突。
- README 保留入口概览，详细工作流写入 `docs/workflows/`。

## 四、测试与验收

- 使用标准库 `unittest` 覆盖索引解析、目录解析、分片、样例配置、图像预处理、运行清单和 montage。
- 运行 `git diff --check`、`python -m compileall`、`ruff check` 和 `bash -n`。
- 检查所有 CLI 的 `--help` 与无需 GPU 的 `--dry-run`。
- 验证构建元数据中的版本为 `2.0.0`。
- 若 GPU、依赖及模型可用，执行最小推理冒烟测试；否则记录跳过原因。
- 最终确认未提交内容中不包含空文件、重复图片、缓存、日志、模型或运行输出。

## 五、执行顺序与默认约定

1. 将本计划写入 `docs/uncommitted-code-refactor-plan.md`。
2. 清理临时文件和重复样例，完成目录迁移。
3. 提取推理公共逻辑并修正 Wan 提前停止接口。
4. 重构自定义输入和 Language-Table 工作流。
5. 统一输出清单、恢复机制和结果工具。
6. 修复版本配置、模型配置、忽略规则和文档。
7. 添加测试并完成静态、dry-run 和条件式 GPU 冒烟验收。
8. 汇总最终变更和验证结果。

默认采用已确认的选择：彻底整理未提交入口、保留两组规范化示例、保留但通用化多机 Shell 启动模板；本轮不创建 commit。
