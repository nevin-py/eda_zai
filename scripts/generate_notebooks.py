from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT_DIR / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)


def write_notebook(path: Path, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nbf.write(nb, path)


def build_eda_notebook() -> None:
    cells = [
        nbf.v4.new_markdown_cell(
            "# EDA - NASA CMAPSS FD001\n"
            "This notebook documents data loading, RUL labeling, variance screening, and core exploratory plots."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "\n"
            "ROOT = Path('..').resolve()\n"
            "SRC = ROOT / 'src'\n"
            "if str(SRC) not in sys.path:\n"
            "    sys.path.insert(0, str(SRC))\n"
            "\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from predictive_maintenance import prepare_train_test_with_rul, detect_near_zero_variance_sensors, SENSOR_COLS, RUL_COL\n"
            "\n"
            "sns.set_theme(style='whitegrid')"
        ),
        nbf.v4.new_code_cell(
            "train_df, test_df = prepare_train_test_with_rul(ROOT / 'data', subset='FD001')\n"
            "train_df.shape, test_df.shape"
        ),
        nbf.v4.new_code_cell(
            "dropped = detect_near_zero_variance_sensors(train_df, threshold=1e-4)\n"
            "active = [s for s in SENSOR_COLS if s not in dropped]\n"
            "dropped"
        ),
        nbf.v4.new_code_cell(
            "corr = train_df[[*active, RUL_COL]].corr(numeric_only=True)[RUL_COL].drop(RUL_COL).abs().sort_values(ascending=False)\n"
            "top = corr.head(8).index.tolist()\n"
            "top"
        ),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)\n"
            "sampled = train_df['engine_id'].drop_duplicates().sample(10, random_state=42)\n"
            "for idx, sensor in enumerate(top[:6]):\n"
            "    ax = axes.flatten()[idx]\n"
            "    for _, g in train_df[train_df['engine_id'].isin(sampled)].groupby('engine_id'):\n"
            "        ax.plot(g['cycle'], g[sensor], alpha=0.35)\n"
            "    ax.set_title(sensor)\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "plt.figure(figsize=(8, 4))\n"
            "sns.histplot(train_df[RUL_COL], bins=40, kde=True)\n"
            "plt.title('RUL Distribution')\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "plt.figure(figsize=(12, 10))\n"
            "sns.heatmap(train_df[[*active, RUL_COL]].corr(numeric_only=True), cmap='coolwarm', center=0)\n"
            "plt.title('Sensors vs RUL Correlation')\n"
            "plt.show()"
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "eda.ipynb", cells)


def build_modeling_notebook() -> None:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Modeling - RUL Regression and Failure Classification\n"
            "This notebook runs the training script and inspects saved metrics/models."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import subprocess\n"
            "\n"
            "ROOT = Path('..').resolve()\n"
            "PYTHON = '/home/ariva/.pyenv/versions/3.11.9/bin/python'"
        ),
        nbf.v4.new_code_cell(
            "result = subprocess.run([PYTHON, str(ROOT / 'scripts/train_models.py')], capture_output=True, text=True)\n"
            "print(result.stdout)\n"
            "print(result.stderr)"
        ),
        nbf.v4.new_code_cell(
            "metrics = json.loads((ROOT / 'models/metrics.json').read_text())\n"
            "metrics"
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "modeling.ipynb", cells)


def build_explainability_notebook() -> None:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Explainability - SHAP for RUL Predictions\n"
            "This notebook executes SHAP artifact generation and previews saved metadata."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import subprocess\n"
            "\n"
            "ROOT = Path('..').resolve()\n"
            "PYTHON = '/home/ariva/.pyenv/versions/3.11.9/bin/python'"
        ),
        nbf.v4.new_code_cell(
            "result = subprocess.run([PYTHON, str(ROOT / 'scripts/run_explainability.py')], capture_output=True, text=True)\n"
            "print(result.stdout)\n"
            "print(result.stderr)"
        ),
        nbf.v4.new_code_cell(
            "meta = json.loads((ROOT / 'models/explainability_metadata.json').read_text())\n"
            "meta"
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "explainability.ipynb", cells)


def main() -> int:
    build_eda_notebook()
    build_modeling_notebook()
    build_explainability_notebook()
    print("Generated notebooks:")
    print("- notebooks/eda.ipynb")
    print("- notebooks/modeling.ipynb")
    print("- notebooks/explainability.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
