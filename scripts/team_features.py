"""
Team feature engineering helpers for Pokemon team battle models.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import pandas as pd

from pokemon_types import calculate_type_advantage, TYPE_EFFECTIVENESS

STAT_COLUMNS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
TYPE_LIST = sorted(TYPE_EFFECTIVENESS.keys())
TEAM_MEMBER_COLUMNS = ["0", "1", "2", "3", "4", "5"]


def load_team_rosters(teams_csv: str) -> pd.DataFrame:
    teams = pd.read_csv(teams_csv)
    teams = teams.rename(columns={"#": "team_id"})
    return teams


def build_team_type_map(
    pokemon_df: pd.DataFrame, teams_df: pd.DataFrame
) -> Dict[int, List[Tuple[str, str | None]]]:
    pokemon_lookup = pokemon_df.set_index("#")
    team_types: Dict[int, List[Tuple[str, str | None]]] = {}

    for _, row in teams_df.iterrows():
        team_id = int(row["team_id"])
        pokemon_ids = [int(row[col]) for col in TEAM_MEMBER_COLUMNS]
        members = pokemon_lookup.loc[pokemon_ids]
        types: List[Tuple[str, str | None]] = []
        for _, member in members.iterrows():
            type1 = str(member["Type 1"])
            type2 = None if pd.isna(member["Type 2"]) else str(member["Type 2"])
            types.append((type1, type2))
        team_types[team_id] = types

    return team_types


def compute_team_type_advantage(
    team_a: Iterable[Tuple[str, str | None]],
    team_b: Iterable[Tuple[str, str | None]],
) -> float:
    total = 0.0
    count = 0
    for a_type1, a_type2 in team_a:
        for b_type1, b_type2 in team_b:
            total += calculate_type_advantage(a_type1, a_type2, b_type1, b_type2)
            count += 1
    return total / count if count else 0.0


def build_team_features(
    pokemon_df: pd.DataFrame, teams_df: pd.DataFrame
) -> pd.DataFrame:
    pokemon_lookup = pokemon_df.set_index("#")
    feature_rows: List[Dict[str, float | int]] = []

    for _, row in teams_df.iterrows():
        team_id = int(row["team_id"])
        pokemon_ids = [int(row[col]) for col in TEAM_MEMBER_COLUMNS]
        members = pokemon_lookup.loc[pokemon_ids]

        features: Dict[str, float | int] = {"team_id": team_id}

        for stat in STAT_COLUMNS:
            stat_values = members[stat].astype(float)
            features[f"{stat}_sum"] = float(stat_values.sum())
            features[f"{stat}_mean"] = float(stat_values.mean())

        features["Legendary_count"] = int(members["Legendary"].astype(int).sum())

        type_counts = {f"Type_count_{t}": 0 for t in TYPE_LIST}
        type_unique: set[str] = set()

        for _, member in members.iterrows():
            type1 = str(member["Type 1"])
            type_counts[f"Type_count_{type1}"] += 1
            type_unique.add(type1)

            if pd.notna(member["Type 2"]):
                type2 = str(member["Type 2"])
                type_counts[f"Type_count_{type2}"] += 1
                type_unique.add(type2)

        features.update(type_counts)
        features["Type_unique_count"] = len(type_unique)

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


def build_matchup_features(
    battles_df: pd.DataFrame,
    team_features: pd.DataFrame,
    team_types: Dict[int, List[Tuple[str, str | None]]],
) -> tuple[pd.DataFrame, pd.Series, List[str]]:
    merged = battles_df.merge(
        team_features, left_on="first", right_on="team_id", suffixes=("", "_A")
    ).merge(
        team_features, left_on="second", right_on="team_id", suffixes=("_A", "_B")
    )

    base_columns = [c for c in team_features.columns if c != "team_id"]
    feature_columns: List[str] = []

    for col in base_columns:
        merged[f"{col}_diff"] = merged[f"{col}_A"] - merged[f"{col}_B"]
        feature_columns.append(f"{col}_diff")

    merged["Team_type_advantage"] = merged.apply(
        lambda row: compute_team_type_advantage(
            team_types[int(row["first"])], team_types[int(row["second"])]
        ),
        axis=1,
    )
    feature_columns.append("Team_type_advantage")

    X = merged[feature_columns]
    y = merged["winner"].astype(int)
    return X, y, feature_columns


def build_single_matchup_features(
    team_a_id: int,
    team_b_id: int,
    team_features: pd.DataFrame,
    team_types: Dict[int, List[Tuple[str, str | None]]],
    feature_columns: List[str],
) -> pd.DataFrame:
    features = {}
    team_a = team_features[team_features["team_id"] == team_a_id]
    team_b = team_features[team_features["team_id"] == team_b_id]

    if team_a.empty or team_b.empty:
        raise ValueError("Team id not found in team features.")

    team_a_row = team_a.iloc[0]
    team_b_row = team_b.iloc[0]

    base_columns = [c for c in team_features.columns if c != "team_id"]
    for col in base_columns:
        features[f"{col}_diff"] = team_a_row[col] - team_b_row[col]

    features["Team_type_advantage"] = compute_team_type_advantage(
        team_types[team_a_id], team_types[team_b_id]
    )

    return pd.DataFrame([features])[feature_columns]
