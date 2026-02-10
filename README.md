# 🎮 Pokemon Battle Predictor

Predict Pokemon battle outcomes using machine learning! Features a desktop GUI with search, type matchup analysis, and win probability predictions.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model** (first time only):
   ```bash
   python scripts/train_model.py
   ```

3. **Launch the GUI:**
   - **Windows:** Double-click `start_gui.bat`
   - **Any OS:** `python run.py`

## 📋 Features

- **Interactive GUI** - Easy-to-use desktop interface
- **Search** - Type to find Pokemon instantly
- **Smart Predictions** - ML model trained on 50,000+ battles
- **Type Advantages** - Detailed offensive matchup analysis
- **Battle Stats** - Compare HP, Attack, Defense, Speed, and more

## 🔧 Advanced Usage

**Command-line predictions:**
```bash
python scripts/predict.py "Pikachu" "Charmander"
```

**Generate random battles:**
```bash
python scripts/generate_battles.py --count 10
```

**Test model accuracy:**
```bash
python scripts/test_model.py
```

## 📁 Project Structure

```
pokemon-battle-prediction/
├── datasets/          # Pokemon and battle data
├── scripts/           # Training, prediction, GUI
├── types/             # Type advantage logic
└── start_gui.bat      # Quick launcher (Windows)
```