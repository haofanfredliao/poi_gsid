## Plan: Notebook → Python 代码迁移

**TL;DR**：将散落在 notebook 中的代码系统性地迁移至 repo 的 py 结构。分 4 个阶段：修复现有 bug → 补全空文件/缺失函数 → 清理硬编码路径 → 补齐基础设施。

---

### Phase 1 — 修复已迁移代码中的 Bug

**1.1 layers.py — 模块级副作用**
- 问题：`config = Config()`、`print(config)`、`os.makedirs(config.ckpt_dir)` 写在模块顶层，每次 `import rqvae.layers` 都会执行，这是错误的
- 修复：将这三行移入 `if __name__ == "__main__":` 块或直接删除（由 rqvae_training.py 负责）

**1.2 trainer.py — `MultiModalContrastiveLoss` 未定义**
- 问题：trainer.py 中 `self.contrastive_loss = MultiModalContrastiveLoss(...)` 引用了一个在任何 .py 文件中都不存在的类
- 来源：定义在 notebooks/multimodal1 (1).ipynb.ipynb) 第3节（`# === 3. 对比学习损失 ===`）
- 修复：将 `MultiModalContrastiveLoss` 追加到 layers.py 末尾，并在 trainer.py 顶部 import

**1.3 poi_dataloader.py — GCS 分支 undefined 变量**
- 问题：`is_gcs=True` 分支中 `PROJECT_ID` 和 `GCS_RELATIVE` 从未定义
- 修复：在 `POIDataset.__init__` 中改为接收 `project_id` 和 `gcs_bucket_path` 参数，或添加清晰的 `raise ValueError` 提示用户配置

---

### Phase 2 — 补全空文件和缺失函数

**2.1 inference.py — 完全为空**
- 从 rqvae1.ipynb Cell 12 迁移 `generate_and_save_semantic_ids(model, dataset, config, save_path)` 
- 添加 `if __name__ == "__main__":` 入口（从 checkpoint 加载模型并调用）

**2.2 embedding_eval.py — 缺失多个分析函数**
- 已有：`plot_training_history`, `evaluate_tsne`
- 待从 rqvae1.ipynb 迁移：
  - `validate_reconstruction()` (Cell 13) — 重构质量验证，MSE/MAE/cosine sim
  - `analyze_codebook_usage()` (Cell 14) — codebook 利用率分布图
  - `save_final_model()` (Cell 15) — 保存模型 + `config.json` + `training_summary.txt`
  - `plot_semantic_clusters()` (Cell 25) — PCA 可视化 + semantic ID 前缀匹配着色

---

### Phase 3 — 路径与配置清理

**3.1 layers.py Config** — `embedding_pkl_path` 默认值改为空字符串，注释说明需用户传入

**3.2 inference.py** — `__main__` 中的 6 个 GCP 绝对路径改为注释说明（parallel with 3.1）

**3.3 embedding_training.py** — 硬编码 `ROOT_DIR`、`image_base_path`、`checkpoint` 路径改为注释或 argparse

---

### Phase 4 — 基础设施补齐（*parallel with all phases above*）

**4.1 `__init__.py`** — `rqvae/`、`embedding/`、`dataset/`、`evaluations/` 四个目录均缺失，需新建空文件以支持正确的 package import

**4.2 pyproject.toml** — 补充缺少的 `safetensors` 依赖（layers.py 和 inference.py 均使用了 `safetensors.torch.load_file`）

---

### Relevant Files

| 文件 | 状态 | 操作 |
|------|------|------|
| layers.py | 有 bug | Phase 1.1 |
| inference.py | **空文件** | Phase 2.1 |
| trainer.py | 基本完整 | 无需改动 |
| rqvae_training.py | 基本完整 | Phase 3.1 路径注释 |
| layers.py | 缺函数 | Phase 1.2 添加 ContrastiveLoss |
| trainer.py | 有 ImportError | Phase 1.2 修复 import |
| inference.py | 完整但路径硬编码 | Phase 3.2 |
| embedding_training.py | 路径硬编码 | Phase 3.3 |
| poi_dataloader.py | 有 bug | Phase 1.3 |
| embedding_eval.py | 不完整 | Phase 2.2 |
| pyproject.toml | 缺依赖 | Phase 4.2 |
| 4×`__init__.py` | **不存在** | Phase 4.1 新建 |

---

### Verification

1. `python -c "from rqvae.layers import RQVAE, Config"` — 不应有任何 print 输出或文件创建
2. `python -c "from embedding.trainer import POIContrastiveTrainer"` — 不应有 `ImportError`
3. `python -c "from rqvae.inference import generate_and_save_semantic_ids"` — 可正常导入
4. `python -c "from evaluations.embedding_eval import validate_reconstruction, analyze_codebook_usage, save_final_model"` — 可正常导入
5. 用 `sample500_preprocessed.csv` 构造一个小型 `.pkl` 测试文件，跑通 rqvae_training.py 的完整训练流程

---

### Decisions

- **不改动逻辑**：只做结构性迁移，不重构已有功能
- **GCS 支持保留**：`is_gcs` 参数保留，但修复 undefined 变量问题
- **rqvae (2).ipynb 后半段的 HTML 报告、case study bucket 展示** 不迁移为 py 脚本（属于 exploratory analysis，保留在 notebook 中更合适）
- **scope 外**：不合并两个 rqvae notebook 的内容差异，不添加 CLI argparse

---

