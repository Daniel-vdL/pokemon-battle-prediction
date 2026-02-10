"""
Team Battle Predictor - Train ML model for 6v6 team battles
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'types'))

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from pokemon_types import calculate_type_advantage


class TeamBattlePredictor:
    """Train and evaluate team battle prediction models."""
    
    def __init__(self):
        self.pokemon = pd.DataFrame()
        self.teams = pd.DataFrame()
        self.combats = pd.DataFrame()
        self.model = None
        
    def load_data(self):
        """Load Pokemon, team, and combat datasets."""
        print("📊 Loading data...")
        base_path = Path(__file__).parent.parent / "datasets"
        
        self.pokemon = pd.read_csv(base_path / "pokemon.csv")
        self.teams = pd.read_csv(base_path / "pokemon_id_each_team.csv")
        self.combats = pd.read_csv(base_path / "team_combat.csv")
        
        print(f"Pokemon: {len(self.pokemon)}")
        print(f"Teams: {len(self.teams)}")
        print(f"Team battles: {len(self.combats)}")
    
    def get_team_pokemon(self, team_id):
        """Get all 6 Pokemon for a team."""
        team_row = self.teams[self.teams["#"] == team_id]
        if team_row.empty:
            return []
        
        pokemon_ids = team_row.iloc[0][['0', '1', '2', '3', '4', '5']].values
        return [self.pokemon[self.pokemon["#"] == pid].iloc[0] for pid in pokemon_ids 
                if not self.pokemon[self.pokemon["#"] == pid].empty]
    
    def calculate_team_features(self, team_id):
        """Calculate aggregated features for a team."""
        team_pokemon = self.get_team_pokemon(team_id)
        
        if len(team_pokemon) != 6:
            return None
        
        # Aggregate stats
        stats = {
            "avg_hp": np.mean([p["HP"] for p in team_pokemon]),
            "avg_attack": np.mean([p["Attack"] for p in team_pokemon]),
            "avg_defense": np.mean([p["Defense"] for p in team_pokemon]),
            "avg_sp_atk": np.mean([p["Sp. Atk"] for p in team_pokemon]),
            "avg_sp_def": np.mean([p["Sp. Def"] for p in team_pokemon]),
            "avg_speed": np.mean([p["Speed"] for p in team_pokemon]),
            "total_stats": sum([
                p["HP"] + p["Attack"] + p["Defense"] + 
                p["Sp. Atk"] + p["Sp. Def"] + p["Speed"] 
                for p in team_pokemon
            ]),
            "legendary_count": sum([1 for p in team_pokemon if p["Legendary"]]),
            "max_hp": max([p["HP"] for p in team_pokemon]),
            "max_attack": max([p["Attack"] for p in team_pokemon]),
            "max_speed": max([p["Speed"] for p in team_pokemon]),
        }
        
        return stats
    
    def prepare_features(self):
        """Prepare training features."""
        print("\n🔄 Preparing features...")
        
        features_list = []
        labels = []
        
        for idx, row in self.combats.iterrows():
            team1_id = row["first"]
            team2_id = row["second"]
            winner = row["winner"]
            
            team1_features = self.calculate_team_features(team1_id)
            team2_features = self.calculate_team_features(team2_id)
            
            if team1_features is None or team2_features is None:
                continue
            
            # Calculate differences
            diff_features = {
                "hp_diff": team1_features["avg_hp"] - team2_features["avg_hp"],
                "attack_diff": team1_features["avg_attack"] - team2_features["avg_attack"],
                "defense_diff": team1_features["avg_defense"] - team2_features["avg_defense"],
                "sp_atk_diff": team1_features["avg_sp_atk"] - team2_features["avg_sp_atk"],
                "sp_def_diff": team1_features["avg_sp_def"] - team2_features["avg_sp_def"],
                "speed_diff": team1_features["avg_speed"] - team2_features["avg_speed"],
                "total_diff": team1_features["total_stats"] - team2_features["total_stats"],
                "legendary_diff": team1_features["legendary_count"] - team2_features["legendary_count"],
                "max_hp_diff": team1_features["max_hp"] - team2_features["max_hp"],
                "max_attack_diff": team1_features["max_attack"] - team2_features["max_attack"],
                "max_speed_diff": team1_features["max_speed"] - team2_features["max_speed"],
            }
            
            features_list.append(diff_features)
            labels.append(winner)
            
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(self.combats)} battles...")
        
        self.X = pd.DataFrame(features_list)
        self.y = pd.Series(labels)
        
        print(f"✅ Prepared {len(self.X)} training examples")
    
    def train_model(self):
        """Train the team battle prediction model."""
        print("\n🤖 Training model...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Try multiple models
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        
        best_accuracy = 0
        best_model = None
        best_name = ""
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
            
            print(f"  Training accuracy: {train_acc:.4f}")
            print(f"  Test accuracy: {test_acc:.4f}")
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_model = model
                best_name = name
        
        self.model = best_model
        print(f"\n✅ Best model: {best_name} (Accuracy: {best_accuracy:.4f})")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test)
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Team 2 Wins", "Team 1 Wins"]))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print("\n🎯 Top 5 Important Features:")
            importances = pd.DataFrame({
                'feature': self.X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(importances.head(5).to_string(index=False))
    
    def save_model(self):
        """Save the trained model."""
        print("\n💾 Saving model...")
        
        model_data = {
            "model": self.model,
            "pokemon": self.pokemon,
            "teams": self.teams,
        }
        
        output_path = Path(__file__).parent.parent / "pokemon_team_model.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {output_path}")


def main():
    predictor = TeamBattlePredictor()
    predictor.load_data()
    predictor.prepare_features()
    predictor.train_model()
    predictor.save_model()
    
    print("\n🎉 Team battle model training complete!")
    print("You can now predict team battles using the GUI or predict script.")


if __name__ == "__main__":
    main()
