# Pytest AI Tests

An example project for testing AI models using Pytest in Python.

---

## Project Goals

Build a simple pipeline to:

- Train a binary classification model using LogisticRegression (from Scikit-learn)
- Preprocess data (normalization)
- Automatically test both components using Pytest

---

## Project Structure

```
pytest-ai-tests/
│
├── README.md                # Project info
├── requirements.txt         # Project dependencies
├── .github/
│   └── workflows/
│       └── python-app.yml   # GitHub Actions CI pipeline
├── src/                     # Source code
│   ├── __init__.py
│   ├── model.py             # ML training and prediction logic
│   └── preprocess.py        # Data normalization functions
├── tests/                   # Pytest unit tests
│   ├── __init__.py
│   ├── test_model.py        # Tests for model training and prediction
│   └── test_preprocess.py   # Tests for data normalization
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/pytest-ai-tests.git
cd pytest-ai-tests
```

### 2. Create a virtual environment

#### On Windows (CMD):
```bash
python -m venv venv
.\venv\Scripts\activate
```

#### On Git Bash / macOS / Linux:
```bash
python -m venv venv
source venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist yet, install manually:

```bash
pip install pytest scikit-learn numpy
```

Then freeze dependencies:

```bash
pip freeze > requirements.txt
```

---

## Run Tests

With the virtual environment activated, run:

```bash
pytest -v tests/
```

Use the `-s` flag to see `print()` outputs (used in test_preprocess.py):

```bash
pytest -s tests/
```

---

## Expected Results

When running the tests, the following outcomes are expected:

### Model Tests (`test_model.py`)

- `train_model()` returns a trained LogisticRegression model with non-null coefficients.
- `predict()` returns predictions with the same length as the input data.

### Preprocessing Tests (`test_preprocess.py`)

- `normalize_data()` standardizes the features so that:
  - Each column has a mean of approximately 0.
  - Each column has a standard deviation of approximately 1.
- These conditions are verified with `assert` statements.

### Example Output

```
tests/test_model.py ..                             # 2 model tests passed

tests/test_preprocess.py Original Data:
[[1 2]
 [3 4]
 [5 6]]

Normalized Data:
[[-1.22474487 -1.22474487]
 [ 0.          0.        ]
 [ 1.22474487  1.22474487]]

Mean of columns (should be ~0):
[0. 0.]
Standard deviation of columns (should be ~1):
[1. 1.]
.
```

Final summary:
```
3 passed in X.XXs
```

---

## GitHub Actions

This project includes a GitHub Actions workflow in `.github/workflows/python-app.yml` that runs all tests automatically on each push or pull request.

---

## Author

Developed as a practice project for a QA role in an AI team.
