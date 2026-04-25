# EditSR Benchmark Framework

EditSR is a symbolic-regression editing framework built on top of a pre-trained sequence model and a repair module. This repository contains the training entry point, benchmark evaluation scripts, and project-relative path utilities required to run the code on a new machine.

## 📥 Pre-trained Weights

**Crucial Step:** Before running any training or evaluation script, you must download the pre-trained model weights.

1. **Download Link:** [Get the Pre-trained Weights Here](https://drive.google.com/file/d/1E2GknLDz5_UZlzlLTe7BOvEGOtDwfsJU/view?usp=sharing)
2. Place the downloaded checkpoint under `scripts/Exp_weights/`.
3. Update `scripts/config.yaml` so that `model_path` points to your local checkpoint, for example:

```yaml
model_path: Exp_weights/Weight.ckpt
```

## ⚠️ Configuration & Paths (Important)

> **Note:** The code uses project-relative defaults where possible, but you must still update local checkpoint, metadata, and dataset paths to match your environment before execution.

**Please manually check and update the paths in the following files:**

1. **`scripts/config.yaml`**:
   * Update `train_path`, `val_path`, and `test_path` if your generated train / validation / test data are stored outside the default `scripts/data/` directory.
   * Update `model_path` to point to your downloaded checkpoint.
   * Check device-related settings under `inference`, especially `device`.

2. **Evaluation scripts (`scripts/evaluate_*.py`)**:
   * **Metadata Path:** The evaluation scripts load vocabulary / metadata from `scripts/data/val_data/10vars/100`. If your metadata is stored elsewhere, search for `metadata_path` and update the path.
   * **Dataset Paths:** Refer to the "Dataset Preparation" section below for the specific variable names to update in each script.

> The evaluation scripts print results directly to stdout by default.


## ▶️ Training

Run training from the project root:

```bash
python scripts/train.py
```

Before training, verify that `scripts/config.yaml` points to the correct train / validation / test datasets and checkpoint location.

## 🧪 Generating Training Data

To generate synthetic symbolic-regression training data, please refer to the implementation and data-generation pipeline from **Neural Symbolic Regression That Scales**:

* [NeuralSymbolicRegressionThatScales Repository](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales)

If you use this data-generation procedure, please cite:

```bibtex
@inproceedings{biggio2021neural,
  title={Neural symbolic regression that scales},
  author={Biggio, Luca and Bendinelli, Tommaso and Neitz, Alexander and Lucchi, Aurelien and Parascandolo, Giambattista},
  booktitle={International conference on machine learning},
  pages={936--945},
  year={2021},
  organization={Pmlr}
}
```

## 📚 Dataset Preparation

Please download the required datasets (`.txt`, `.tsv`, `.tsv.gz`, or `.csv`) from the official repositories or standardized benchmark collections listed below. After downloading, you **must update the file paths** in the corresponding evaluation scripts.

### 1. AI Feynman Dataset

* **Primary Source:** *AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity* (Udrescu et al., 2020).
* **Alternative Source (Standardized):** *Contemporary symbolic regression methods and their relative performance* (La Cava et al., 2021).
* **Download:**
  * [Original AI Feynman Repo](https://github.com/SJ001/AI-Feynman)
  * [SRBench / PMLB Repository](https://github.com/cavalab/srbench)
* **Target Script:** `scripts/evaluate_feynman.py`
* **Configuration:** Update the dataset root and label file paths:

```python
# Variable: directory
directory = str(scripts_path("Feynman_with_units"))

# Variable: directory2
directory2 = str(scripts_path("FeynmanEquations.xlsx"))
```

Expected local assets:

```text
scripts/Feynman_with_units/
scripts/FeynmanEquations.xlsx
```

### 2. ODE-Strogatz Dataset

* **Primary Source:** *Nonlinear dynamics and chaos* (Strogatz, 2018).
* **Digital Source (Standardized):** *Contemporary symbolic regression methods and their relative performance* (La Cava et al., 2021).
* **Download:** Please acquire the Strogatz ODE benchmark dataset, commonly available via the [SRBench Repository](https://github.com/cavalab/srbench).
* **Target Script:** `scripts/evaluate_ode_strogatz.py`
* **Configuration:** Update the dataset directory and label file:

```python
# Variable: directory
directory = str(scripts_path("ode-strogatz-master", "ode-strogatz-master"))

# Variable: directory2
directory2 = str(scripts_path("ode-strogatz-master", "ode.xlsx"))
```

Expected local assets:

```text
scripts/ode-strogatz-master/ode-strogatz-master/
scripts/ode-strogatz-master/ode.xlsx
```

### 3. SRBench / Black-box Datasets

* **Source:** *Call for Action: towards the next generation of symbolic regression benchmark* (Aldeia et al., 2025) and the SRBench / PMLB benchmark collection.
* **Download:** [SRBench / PMLB Repository](https://github.com/cavalab/srbench). Look for the black-box regression datasets.
* **Target Script:** `scripts/evaluate_black_box.py`
* **Configuration:** Update the local data directory if you are not using the default path:

```python
# Variable: LOCAL_DATA_DIR
LOCAL_DATA_DIR = str(scripts_path("srbench_blackbox_datasets"))
```

Expected local assets:

```text
scripts/srbench_blackbox_datasets/
```

The script first tries to load datasets through `pmlb`; if unavailable, it falls back to local CSV files in `LOCAL_DATA_DIR`.

### 4. First-Principles Datasets

* **Source:** *Contemporary symbolic regression methods and their relative performance* (La Cava et al., 2021).
* **Download:** [SRBench / PMLB Repository](https://github.com/cavalab/srbench). Look for first-principles datasets, for example `first_principles_bode`.
* **Target Script:** `scripts/evaluate_first_principles.py`
* **Configuration:** Update the root path for first-principles datasets:

```python
# Variable: DATASETS_ROOT
DATASETS_ROOT = os.path.join(SCRIPT_DIR, "datasets_first", "firstprinciples")
```

Expected local assets:

```text
scripts/datasets_first/firstprinciples/
```

### 5. Repair-trace Benchmark Table

* **Target Script:** `scripts/evaluate_repair_trace.py`
* **Configuration:** The script expects a benchmark table with columns `name`, `variables`, `expression`, `train_range`, and `test_range`.

```python
# Default table path
scripts/table3_with_n200.csv
```

Expected local asset:

```text
scripts/table3_with_n200.csv
```

### 6. Distractor-robustness Benchmark Table

* **Target Script:** `scripts/evaluate_distractor_robustness.py`
* **Configuration:** The script expects a benchmark table with columns `name`, `variables`, `expression`, `train_range`, and `test_range`.

```python
# Default table path
scripts/table3_with_n200_1.csv
```

Expected local asset:

```text
scripts/table3_with_n200_1.csv
```

## 🧩 Repository Notes

* `src/EditSR/project_paths.py` centralizes project-relative path resolution.
* `scripts/config.yaml` controls training, evaluation, architecture, repair-head, inference, and BFGS settings.
* `scripts/data/val_data/10vars/100` stores the default validation metadata used by evaluation scripts.
* Generated train / validation data should follow the directory structure configured by `train_path`, `val_path`, and `test_path` in `scripts/config.yaml`.

## 📁 Suggested Repository Layout

```text
EditSR/
├── scripts/
│   ├── config.yaml
│   ├── train.py
│   ├── evaluate_feynman.py
│   ├── evaluate_ode_strogatz.py
│   ├── evaluate_black_box.py
│   ├── evaluate_first_principles.py
│   ├── evaluate_repair_trace.py
│   ├── evaluate_distractor_robustness.py
│   ├── Exp_weights/
│   │   └── Weight.ckpt
│   ├── data/
│   │   ├── train_data/
│   │   └── val_data/
│   ├── Feynman_with_units/
│   ├── ode-strogatz-master/
│   ├── srbench_blackbox_datasets/
│   └── datasets_first/firstprinciples/
└── src/EditSR/
```

## References

1. Udrescu, S. M., et al. (2020). "AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity". *Advances in Neural Information Processing Systems*.
2. Strogatz, S. H. (2018). *Nonlinear dynamics and chaos: with applications to physics, biology, chemistry, and engineering*. CRC Press.
3. La Cava, W., et al. (2021). "Contemporary symbolic regression methods and their relative performance". *arXiv preprint arXiv:2107.14351*.
4. Aldeia, G. S. I., et al. (2025). "Call for Action: towards the next generation of symbolic regression benchmark". *arXiv preprint arXiv:2505.03977*.
5. Biggio, L., Bendinelli, T., Neitz, A., Lucchi, A., & Parascandolo, G. (2021). "Neural symbolic regression that scales". *International Conference on Machine Learning*, 936--945.
