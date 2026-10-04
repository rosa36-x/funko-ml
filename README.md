# funko-ml

ML mini-project (UE24CS352A): predicting the secondary-market price range of Funko Pop collectibles, reproducing a published paper and extending it with XGBoost.

Reference paper: *Application of Machine Learning Methods to Evaluate Collectibles* (Xiaotian Lu, CS229 project, https://github.com/timothyxlu/CS229/).

## What this project does

**Task.** Given four facts about a Funko Pop (franchise category, material finish, special/exclusive edition, and online search volume), predict which of 5 price bins its average resale price falls in. Bins are 5 equal-width slices of log price, so bin 1 is the cheapest and bin 5 the most expensive. The dataset has 359 items, and the bins are very imbalanced (146 / 153 / 46 / 10 / 4 items).

**Part 1, reproduction.** Rebuild the paper's models on the same data, folds and metrics: linear SVM, SVM with a custom kernel for mixed features, decision tree, and bagged decision tree.

**Part 2, extension.** Add one model the paper does not use (XGBoost) and compare it with the paper's models under identical conditions.

Every model is scored with the same code (`src/evaluate.py`) on the same 4 stratified folds (`data/folds.pkl`), so results are directly comparable.

## Repository layout

```
funko-ml/
├── data/
│   ├── data_curated.csv        raw curated dataset (359 rows)
│   ├── processed_table.csv     dataset plus price_bin and normalised search volume
│   ├── processed.npz           feature matrices and labels (X_onehot, X_ordinal, y, y0)
│   └── folds.pkl               the 4 saved stratified train/validation splits
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_linear_svm.ipynb              linear SVM and RBF SVM
│   ├── 03_custom_kernel_svm.ipynb       paper's custom kernel
│   ├── 04_decision_tree.ipynb
│   ├── 05_bagged_decision_tree.ipynb    sweep over number of trees
│   ├── 06_xgboost.ipynb                 extension model
│   └── 07_compare.ipynb                 reported vs ours; XGBoost vs the rest
├── src/evaluate.py             shared data loading and metric code
├── results/                    one CSV per model, plus comparison tables
├── requirements.txt
├── slides/  writeup/
```

## Notebooks and run order

| # | Notebook | What it does | Needs |
|---|---|---|---|
| 01 | Data preprocessing | Builds price bins, normalises search volume, builds one-hot and ordinal feature matrices, creates and saves the folds | `data/data_curated.csv` |
| 02 | Linear SVM | Linear and RBF SVM baselines | files from 01 |
| 03 | Custom-kernel SVM | Implements the paper's kernel and trains an SVM on the precomputed kernel | files from 01 |
| 04 | Decision tree | Fully grown tree, `random_state=1` | files from 01 |
| 05 | Bagged decision tree | Sweeps 2 to 200 trees | files from 01 |
| 06 | XGBoost | Default, class-weighted and shallow variants | files from 01 |
| 07 | Compare | Paper numbers vs ours, then XGBoost vs the paper's models | CSVs from 02 to 06 |

`data/processed.npz`, `data/folds.pkl` and the `results/` CSVs are committed, so you can open any single notebook without re-running the earlier ones. Re-run in order (01 to 07) if you want to regenerate everything.

## Running on Google Colab (what we used)

1. Open the notebook in Colab (upload it, or open it from GitHub via File > Open notebook > GitHub).
2. Get the repo into the Colab session, using whichever the notebook's first cells expect:
   - Notebooks 05, 06 and 07 clone the repo themselves with `!git clone https://github.com/rosa36-x/funko-ml.git` followed by `%cd funko-ml/notebooks`. Just run those cells.
   - Notebooks 02, 03 and 04 unzip `/content/funko-ml-main.zip`. On GitHub choose Code > Download ZIP (the file is named `funko-ml-main.zip`) and upload it to the Colab file panel (the folder icon on the left) before running the first cell.
3. Run all cells from the top (Runtime > Run all).
4. Colab already ships with numpy, pandas, scikit-learn, matplotlib and XGBoost. If `import xgboost` fails, run `!pip install xgboost` in a cell.
5. Cells ending in `files.download(...)` download the results to your computer. Colab sessions are temporary, so download or commit the outputs you want to keep.

## Running locally

Requirements: Python 3.10 or newer, and scikit-learn 1.2 or newer (the code uses the `sparse_output` and `estimator=` argument names introduced in 1.2).

```bash
git clone https://github.com/rosa36-x/funko-ml.git
cd funko-ml

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook                   # open the notebooks/ folder
```

Always run notebooks with their own folder (`notebooks/`) as the working directory. They use relative paths such as `../data`, `../src` and `../results`. Jupyter does this by default when you open a notebook from the `notebooks/` folder.

**Colab-only cells.** Notebooks 02 to 07 were written for Colab. Locally, change or skip the cells below. Notebook 01 runs unchanged.

| Notebook | Skip | Edit |
|---|---|---|
| 02 | the first cell (`!unzip ...`) and `%cd /content/...` | the `src_path = '/content/funko-ml-main/src'` line: use `'../src'` |
| 03 | the first cell (`!unzip ...`) | `SRC` and `evaluate_path`: use `'../src'` and `'../src/evaluate.py'` |
| 04 | the first cell (`!unzip ...`) | `evaluate_path`, the `data_dir=` argument of `load_data` (use `'../data'`), and `result_path` (use `'../results/decision_tree.csv'`) |
| 05, 06, 07 | the `!git clone ...` / `%cd ...` cell (and the `!rm` lines in 05) | nothing |
| 02 to 07 | the final cells that import `google.colab` and call `files.download` | nothing |

After those edits, run the notebooks top to bottom. In Jupyter, use Kernel > Restart & Run All.
