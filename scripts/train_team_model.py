"""
Pokemon Team Battle Predictor - AI Training System
Train machine learning models to predict team vs team outcomes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.base import ClassifierMixin

# Make types/ available for import
sys.path.append(str(Path(__file__).parent.parent / "types"))

from team_features import (
    build_matchup_features,
    build_team_features,
    build_team_type_map,
    load_team_rosters,
)


class TeamBattlePredictor:
    """Class for training team battle prediction models."""

    def __init__(
        self,
        pokemon_csv: str = "datasets/pokemon.csv",
        teams_csv: str = "datasets/pokemon_id_each_team.csv",
        battles_csv: str = "datasets/team_combat.csv",
    ) -> None:
        self.pokemon_csv = pokemon_csv
        self.teams_csv = teams_csv
        self.battles_csv = battles_csv

        self.pokemon: pd.DataFrame = pd.DataFrame()
        self.teams: pd.DataFrame = pd.DataFrame()
        self.battles: pd.DataFrame = pd.DataFrame()

        self.team_features: pd.DataFrame = pd.DataFrame()
        self.team_types: Dict[int, List[tuple[str, str | None]]] = {}
        self.feature_columns: List[str] = []

        self.X_train: pd.DataFrame = pd.DataFrame()
        self.X_test: pd.DataFrame = pd.DataFrame()
        self.y_train: pd.Series = pd.Series(dtype=int)
        self.y_test: pd.Series = pd.Series(dtype=int)

        self.best_model: ClassifierMixin | None = None
        self.best_model_name: str = ""

    def load_data(self) -> None:
        print("\n📊 Loading team data...")
        self.pokemon = pd.read_csv(self.pokemon_csv)
        self.teams = load_team_rosters(self.teams_csv)
        self.battles = pd.read_csv(self.battles_csv)

        print(f"Pokemon: {len(self.pokemon)}")
        print(f"Teams: {len(self.teams)}")
        print(f"Team battles: {len(self.battles)}")

    def prepare_features(self) -> None:
        print("\n🔄 Building team features...")
        self.team_features = build_team_features(self.pokemon, self.teams)
        self.team_types = build_team_type_map(self.pokemon, self.teams)

        X, y, feature_columns = build_matchup_features(
            self.battles, self.team_features, self.team_types
        )

        self.feature_columns = feature_columns
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Features: {len(self.feature_columns)}")
        print(f"Train samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")

    def train_models(self) -> Dict[str, Dict[str, Any]]:
        print("\n🤖 Training models...")
        models: Dict[str, ClassifierMixin] = {
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        }

        results: Dict[str, Dict[str, Any]] = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)

            results[name] = {
                "model": model,
                "accuracy": accuracy,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "predictions": y_pred,
            }

            print(f"  ✓ Test Accuracy: {accuracy:.4f}")
            print(f"  ✓ CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        return results

    def select_best_model(self, results: Dict[str, Dict[str, Any]]) -> None:
        self.best_model_name = max(results, key=lambda x: results[x]["accuracy"])
        self.best_model = results[self.best_model_name]["model"]

        print("\n" + "=" * 60)
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print(
            f"🎯 Accuracy: {results[self.best_model_name]['accuracy']:.4f} "
            f"({results[self.best_model_name]['accuracy']*100:.2f}%)"
        )
        print("=" * 60)

    def show_confusion_matrix(self, results: Dict[str, Dict[str, Any]]) -> None:
        print("\n📈 Confusion Matrix:")
        cm = confusion_matrix(self.y_test, results[self.best_model_name]["predictions"])
        print(cm)

    def save_model(self, filename: str = "team_model.pkl") -> None:
        assert self.best_model is not None, "Model must be trained first"

        print(f"\n💾 Saving model to '{filename}'...")
        model_data = {
            "model": self.best_model,
            "pokemon": self.pokemon,
            "teams": self.teams,
            "team_features": self.team_features,
            "feature_columns": self.feature_columns,
            "model_name": self.best_model_name,
            "accuracy": accuracy_score(
                self.y_test, self.best_model.predict(self.X_test)
            ),
        }

        with open(filename, "wb") as f:
            pickle.dump(model_data, f)

        print("✅ Team model saved!")


def main() -> None:
    print("🔥 Pokemon Team Battle Predictor - Training System 🔥")
    print("=" * 60)

    predictor = TeamBattlePredictor()
    predictor.load_data()
    predictor.prepare_features()

    results = predictor.train_models()
    predictor.select_best_model(results)
    predictor.show_confusion_matrix(results)
    predictor.save_model()

    print("\n💡 Use: python predict.py --team <team_a_id> <team_b_id>")


if __name__ == "__main__":
    main()
