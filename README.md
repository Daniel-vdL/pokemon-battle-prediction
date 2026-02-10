# Pokemon Battle Predictor

Predict Pokemon battle outcomes using machine learning. Includes a desktop GUI, type matchup analysis, and win probability predictions.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model (first time only):
   ```bash
   python scripts/train_model.py
   ```

3. Launch the GUI:
   - Windows: double-click `start_gui.bat`
   - Any OS: `python run.py`

## Usage

### Train the Model

```bash
python scripts/train_model.py
```

### Train the Team Model

```bash
python scripts/train_team_model.py
```

### Predict Battles

```bash
python scripts/predict.py "Pikachu" "Charmander"
```

Or run in interactive mode:
```bash
python scripts/predict.py
```

### Predict Team Battles

```bash
python scripts/predict.py --team 1 2
```

### Launch the GUI

```bash
python scripts/gui_app.py
```

Or use the quick launchers:
```bash
python run.py
```

Windows:
```bash
start_gui.bat
```

### Generate Random Battles

```bash
python scripts/generate_battles.py --count 10
```

### Test Model

```bash
python scripts/test_model.py
```

### Test Team Model

```bash
python scripts/test_team_model.py
```

## Project Structure

```
pokemon-battle-prediction/
├── datasets/           # Pokemon and battle datasets
├── scripts/            # Python scripts
├── types/              # Type advantage calculations
├── requirements.txt    # Python dependencies
├── run.py              # GUI launcher
└── start_gui.bat       # Windows GUI launcher
```