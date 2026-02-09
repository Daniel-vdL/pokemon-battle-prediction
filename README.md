# Pokemon Battle Prediction

AI-powered Pokemon battle outcome predictor using machine learning.

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Train the Model

```bash
python scripts/train_model.py
```

### Predict Battles

```bash
python scripts/predict.py "Pikachu" "Charmander"
```

Or run in interactive mode:
```bash
python scripts/predict.py
```

### Generate Random Battles

```bash
python scripts/generate_battles.py --count 10
```

### Test Model

```bash
python scripts/test_model.py
```

## Project Structure

```
pokemon-battle-prediction/
├── datasets/           # Pokemon and battle datasets
├── scripts/            # Python scripts
├── types/              # Type advantage calculations
├── requirements.txt    # Python dependencies
└── README.md          # This file
```