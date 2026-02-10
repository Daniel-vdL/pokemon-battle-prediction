"""
Pokemon Team Battle Predictor - Quick Validation Test
Quick sanity check for team model performance.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pickle

import pandas as pd

# Make types/ available for import
sys.path.append(str(Path(__file__).parent.parent / "types"))

from team_features import (
    build_matchup_features,
    build_team_features,
    build_team_type_map,
    load_team_rosters,
)


def quick_team_test(sample_size: int = 200, random_state: int = 42) -> None:
    print("⚡ QUICK TEAM MODEL VALIDATION TEST")
    print("=" * 60)

    with open("team_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    pokemon = model_data["pokemon"]
    teams = model_data.get("teams")
    if teams is None:
        teams = load_team_rosters("datasets/pokemon_id_each_team.csv")
    else:
        teams = teams.rename(columns={"#": "team_id"})

    battles = pd.read_csv("datasets/team_combat.csv")

    team_features = model_data.get("team_features", pd.DataFrame())
    if team_features.empty:
        team_features = build_team_features(pokemon, teams)

    team_types = build_team_type_map(pokemon, teams)

    X, y, feature_columns = build_matchup_features(battles, team_features, team_types)
    if "feature_columns" in model_data:
        feature_columns = model_data["feature_columns"]
        X = X[feature_columns]

    total = len(X)
    sample_size = min(sample_size, total)
    sample = X.sample(n=sample_size, random_state=random_state)
    y_sample = y.loc[sample.index]

    preds = model.predict(sample)
    probs = model.predict_proba(sample)[:, 1]

    accuracy = (preds == y_sample).mean()
    avg_confidence = float(pd.Series(probs).mean())

    print(f"Samples: {sample_size}/{total}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Average win probability: {avg_confidence * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    quick_team_test()
