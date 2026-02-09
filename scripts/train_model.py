"""
Pokemon Battle Predictor - AI Training System
Train machine learning modellen om Pokemon gevechten te voorspellen.
"""
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import ClassifierMixin

from pokemon_types import calculate_type_advantage


class PokemonBattlePredictor:
    """Class voor het trainen en evalueren van Pokemon battle prediction modellen."""
    
    def __init__(self, pokemon_csv: str = "datasets/pokemon.csv", 
                 combats_csv: str = "datasets/combats.csv") -> None:
        """
        Initialiseer de battle predictor.
        
        Args:
            pokemon_csv: Path naar Pokemon dataset
            combats_csv: Path naar combats dataset
        """
        self.pokemon_csv = pokemon_csv
        self.combats_csv = combats_csv
        self.pokemon: pd.DataFrame = pd.DataFrame()
        self.combats: pd.DataFrame = pd.DataFrame()
        self.data: pd.DataFrame = pd.DataFrame()
        self.X_train: pd.DataFrame = pd.DataFrame()
        self.X_test: pd.DataFrame = pd.DataFrame()
        self.y_train: pd.Series = pd.Series(dtype=int)
        self.y_test: pd.Series = pd.Series(dtype=int)
        self.best_model: ClassifierMixin | None = None
        self.best_model_name: str = ""
        self.feature_columns: List[str] = []
        
    def load_data(self) -> None:
        """Laad Pokemon en combat datasets."""
        print("📊 Data laden...")
        self.pokemon = pd.read_csv(self.pokemon_csv)
        self.combats = pd.read_csv(self.combats_csv)
        
        print(f"Pokémon dataset: {len(self.pokemon)} pokémon")
        print(f"Combat dataset: {len(self.combats)} gevechten")
    
    def prepare_features(self) -> None:
        """Bereid features voor door Pokemon stats te mergen en verschillen te berekenen."""
        print("\n🔄 Data voorbereiden...")
        
        # Merge Pokemon stats voor beide vechters
        self.data = self.combats.merge(
            self.pokemon, left_on="First_pokemon", right_on="#", suffixes=("", "_A")
        ).merge(
            self.pokemon, left_on="Second_pokemon", right_on="#", suffixes=("_A", "_B")
        )
        
        # Basis stat verschillen
        self.data["HP_diff"] = self.data["HP_A"] - self.data["HP_B"]
        self.data["Attack_diff"] = self.data["Attack_A"] - self.data["Attack_B"]
        self.data["Defense_diff"] = self.data["Defense_A"] - self.data["Defense_B"]
        self.data["Sp_Atk_diff"] = self.data["Sp. Atk_A"] - self.data["Sp. Atk_B"]
        self.data["Sp_Def_diff"] = self.data["Sp. Def_A"] - self.data["Sp. Def_B"]
        self.data["Speed_diff"] = self.data["Speed_A"] - self.data["Speed_B"]
        self.data["Legendary_diff"] = (
            self.data["Legendary_A"].astype(int) - self.data["Legendary_B"].astype(int)
        )
        
        # Totale stats berekenen
        self.data["Total_A"] = (
            self.data["HP_A"] + self.data["Attack_A"] + self.data["Defense_A"] + 
            self.data["Sp. Atk_A"] + self.data["Sp. Def_A"] + self.data["Speed_A"]
        )
        self.data["Total_B"] = (
            self.data["HP_B"] + self.data["Attack_B"] + self.data["Defense_B"] + 
            self.data["Sp. Atk_B"] + self.data["Sp. Def_B"] + self.data["Speed_B"]
        )
        self.data["Total_diff"] = self.data["Total_A"] - self.data["Total_B"]
        
        # Type advantage berekenen
        print("🎯 Type advantages berekenen...")
        self.data["Type_advantage"] = self.data.apply(
            lambda row: calculate_type_advantage(
                str(row["Type 1_A"]),
                str(row["Type 2_A"]) if pd.notna(row["Type 2_A"]) else None,
                str(row["Type 1_B"]),
                str(row["Type 2_B"]) if pd.notna(row["Type 2_B"]) else None
            ),
            axis=1
        )
        
        # Target variable: 1 als First_pokemon wint, 0 als Second_pokemon wint
        self.data["Winner_binary"] = (
            self.data["Winner"] == self.data["First_pokemon"]
        ).astype(int)
        
        print(f"Dataset grootte: {len(self.data)} samples")
    
    def prepare_train_test_split(self, test_size: float = 0.2, 
                                 random_state: int = 42) -> None:
        """
        Splits data in training en test sets.
        
        Args:
            test_size: Fractie van data voor test set
            random_state: Random seed voor reproduceerbaarheid
        """
        # Features selecteren (inclusief type advantage)
        self.feature_columns = [
            "HP_diff", "Attack_diff", "Defense_diff", 
            "Sp_Atk_diff", "Sp_Def_diff", "Speed_diff",
            "Legendary_diff", "Total_diff", "Type_advantage"
        ]
        
        X = self.data[self.feature_columns]
        y = self.data["Winner_binary"]
        
        print(f"\nFeatures gebruikt: {self.feature_columns}")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n📈 Training set: {len(self.X_train)} samples")
        print(f"📊 Test set: {len(self.X_test)} samples")
    
    def train_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Train verschillende ML modellen en evalueer ze.
        
        Returns:
            Dictionary met model resultaten
        """
        print("\n🤖 AI Modellen trainen...")
        print("-" * 60)
        
        models: Dict[str, ClassifierMixin] = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
        }
        
        results: Dict[str, Dict[str, Any]] = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Voorspellingen
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            results[name] = {
                "model": model,
                "accuracy": accuracy,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "predictions": y_pred
            }
            
            print(f"  ✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  ✓ CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return results
    
    def select_best_model(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Selecteer het beste model op basis van accuracy.
        
        Args:
            results: Dictionary met model resultaten
        """
        self.best_model_name = max(results, key=lambda x: results[x]["accuracy"])
        self.best_model = results[self.best_model_name]["model"]
        
        print("\n" + "=" * 60)
        print(f"🏆 BESTE MODEL: {self.best_model_name}")
        print(f"🎯 Accuracy: {results[self.best_model_name]['accuracy']:.4f} "
              f"({results[self.best_model_name]['accuracy']*100:.2f}%)")
        print("=" * 60)
    
    def show_feature_importance(self) -> None:
        """Toon feature importance voor tree-based modellen."""
        if self.best_model_name in ["Random Forest", "Gradient Boosting"]:
            assert self.best_model is not None
            print("\n📊 Feature Importance:")
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.best_model.feature_importances_  # type: ignore
            }).sort_values('importance', ascending=False)
            
            for _, row in feature_importance.iterrows():
                print(f"  {row['feature']:20s}: {row['importance']:.4f}")
    
    def show_confusion_matrix(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Toon confusion matrix voor beste model.
        
        Args:
            results: Dictionary met model resultaten
        """
        print("\n📈 Confusion Matrix:")
        cm = confusion_matrix(self.y_test, results[self.best_model_name]["predictions"])
        print(cm)

    
    def predict_battle(self, pokemon1_name: str, pokemon2_name: str) -> str | None:
        """
        Voorspel welke Pokemon zou winnen in een gevecht.
        
        Args:
            pokemon1_name: Naam van eerste Pokemon
            pokemon2_name: Naam van tweede Pokemon
        
        Returns:
            Naam van voorspelde winnaar of None als Pokemon niet gevonden
        """
        assert self.best_model is not None, "Model moet eerst getraind worden"
        
        # Pokemon zoeken
        p1 = self.pokemon[self.pokemon["Name"] == pokemon1_name]
        p2 = self.pokemon[self.pokemon["Name"] == pokemon2_name]
        
        if p1.empty:
            print(f"❌ Pokémon '{pokemon1_name}' niet gevonden!")
            return None
        if p2.empty:
            print(f"❌ Pokémon '{pokemon2_name}' niet gevonden!")
            return None
        
        p1_data = p1.iloc[0]
        p2_data = p2.iloc[0]
        
        # Features berekenen
        type_advantage = calculate_type_advantage(
            str(p1_data["Type 1"]),
            str(p1_data["Type 2"]) if pd.notna(p1_data["Type 2"]) else None,
            str(p2_data["Type 1"]),
            str(p2_data["Type 2"]) if pd.notna(p2_data["Type 2"]) else None
        )
        
        features = pd.DataFrame({
            "HP_diff": [p1_data["HP"] - p2_data["HP"]],
            "Attack_diff": [p1_data["Attack"] - p2_data["Attack"]],
            "Defense_diff": [p1_data["Defense"] - p2_data["Defense"]],
            "Sp_Atk_diff": [p1_data["Sp. Atk"] - p2_data["Sp. Atk"]],
            "Sp_Def_diff": [p1_data["Sp. Def"] - p2_data["Sp. Def"]],
            "Speed_diff": [p1_data["Speed"] - p2_data["Speed"]],
            "Legendary_diff": [int(p1_data["Legendary"]) - int(p2_data["Legendary"])],
            "Total_diff": [
                (p1_data["HP"] + p1_data["Attack"] + p1_data["Defense"] + 
                 p1_data["Sp. Atk"] + p1_data["Sp. Def"] + p1_data["Speed"]) -
                (p2_data["HP"] + p2_data["Attack"] + p2_data["Defense"] + 
                 p2_data["Sp. Atk"] + p2_data["Sp. Def"] + p2_data["Speed"])
            ],
            "Type_advantage": [type_advantage]
        })
        
        # Voorspelling
        prediction = self.best_model.predict(features)[0]
        probability: float | None = None
        
        # Probabilities (als model dit ondersteunt)
        if hasattr(self.best_model, "predict_proba"):
            probs = self.best_model.predict_proba(features)[0]
            probability = float(probs[1] if prediction == 1 else probs[0])
        
        # Resultaat printen
        self._print_battle_result(p1_data, p2_data, pokemon1_name, pokemon2_name,
                                 prediction, probability, type_advantage)
        
        winner = pokemon1_name if prediction == 1 else pokemon2_name
        return winner
    
    def _print_battle_result(self, p1_data: pd.Series, p2_data: pd.Series,
                           pokemon1_name: str, pokemon2_name: str,
                           prediction: int, probability: float | None,
                           type_advantage: float) -> None:
        """Helper functie om battle resultaat te printen."""
        print("\n" + "=" * 60)
        print(f"⚔️  {pokemon1_name} VS {pokemon2_name} ⚔️")
        print("=" * 60)
        
        # Pokemon 1 stats
        print(f"\n{pokemon1_name}:")
        print(f"  Type: {p1_data['Type 1']}" + 
              (f"/{p1_data['Type 2']}" if pd.notna(p1_data['Type 2']) else ""))
        print(f"  HP: {p1_data['HP']}, Attack: {p1_data['Attack']}, Defense: {p1_data['Defense']}")
        print(f"  Speed: {p1_data['Speed']}, Legendary: {p1_data['Legendary']}")
        
        # Pokemon 2 stats
        print(f"\n{pokemon2_name}:")
        print(f"  Type: {p2_data['Type 1']}" + 
              (f"/{p2_data['Type 2']}" if pd.notna(p2_data['Type 2']) else ""))
        print(f"  HP: {p2_data['HP']}, Attack: {p2_data['Attack']}, Defense: {p2_data['Defense']}")
        print(f"  Speed: {p2_data['Speed']}, Legendary: {p2_data['Legendary']}")
        
        # Type advantage info
        if type_advantage > 0.5:
            print(f"\n🎯 {pokemon1_name} heeft een type voordeel! (+{type_advantage:.2f})")
        elif type_advantage < -0.5:
            print(f"\n🎯 {pokemon2_name} heeft een type voordeel! (+{-type_advantage:.2f})")
        
        # Winnaar
        winner = pokemon1_name if prediction == 1 else pokemon2_name
        print(f"\n🏆 VOORSPELLING: {winner} zou winnen!")
        
        if probability:
            print(f"🎯 Confidence: {probability*100:.1f}%")
        
        print("=" * 60)
    
    def save_model(self, filename: str = "pokemon_model.pkl") -> None:
        """
        Sla het beste model op voor later gebruik.
        
        Args:
            filename: Bestandsnaam voor opgeslagen model
        """
        assert self.best_model is not None, "Model moet eerst getraind worden"
        
        print(f"\n💾 Model opslaan als '{filename}'...")
        model_data = {
            "model": self.best_model,
            "pokemon": self.pokemon,
            "features": self.feature_columns,
            "accuracy": accuracy_score(
                self.y_test,
                self.best_model.predict(self.X_test)
            ),
            "model_name": self.best_model_name
        }
        
        with open(filename, "wb") as f:
            pickle.dump(model_data, f)
        
        print("✅ Model succesvol opgeslagen!")


def main() -> None:
    """Main functie om de battle predictor te trainen."""
    print("🔥 Pokémon Battle Predictor - AI Training System 🔥")
    print("=" * 60)
    
    # Initialiseer en train
    predictor = PokemonBattlePredictor()
    predictor.load_data()
    predictor.prepare_features()
    predictor.prepare_train_test_split()
    
    # Train modellen
    results = predictor.train_models()
    predictor.select_best_model(results)
    predictor.show_feature_importance()
    predictor.show_confusion_matrix(results)
    
    # Voorbeeld gevechten
    print("\n\n🎮 VOORBEELD GEVECHTEN:")
    print("=" * 60)
    
    predictor.predict_battle("Pikachu", "Bulbasaur")
    predictor.predict_battle("Charizard", "Blastoise")
    predictor.predict_battle("Mewtwo", "Dragonite")
    
    # Model opslaan
    predictor.save_model()
    
    print("\n\n💡 Gebruik de functie predict_battle(pokemon1, pokemon2) "
          "om je eigen gevechten te voorspellen!")
    print(f"Beschikbaar: {len(predictor.pokemon)} pokémon in de database")
    print("\n🎮 Of gebruik: python predict.py [pokemon1] [pokemon2]")
    print("   Voorbeeld: python predict.py Pikachu Charizard")


if __name__ == "__main__":
    main()
