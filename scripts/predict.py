"""
Pokemon Battle Predictor - CLI Tool
Interactieve tool om Pokemon gevechten te voorspellen met getraind model.

Gebruik: 
    python predict.py "Pokemon1" "Pokemon2"
    python predict.py  (voor interactieve mode)
"""
from typing import Optional, Dict, Any
import pandas as pd
import pickle
import sys
import os

from pokemon_types import (
    calculate_type_advantage,
    get_offensive_matchup,
    format_type_matchup_info
)


class BattlePredictor:
    """Class voor het voorspellen van Pokemon gevechten met een getraind model."""
    
    def __init__(self, model_path: str = "pokemon_model.pkl") -> None:
        """
        Initialiseer de battle predictor.
        
        Args:
            model_path: Path naar het opgeslagen model
        """
        self.model_path = model_path
        self.model_data: Dict[str, Any] = {}
        self.pokemon: pd.DataFrame = pd.DataFrame()
        self.best_model: Any = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Laad het getrainde model."""
        if not os.path.exists(self.model_path):
            print("⚙️  Model niet gevonden, eerst trainen...")
            print("Run eerst: python example.py")
            sys.exit(1)
        
        with open(self.model_path, "rb") as f:
            self.model_data = pickle.load(f)
        
        self.best_model = self.model_data["model"]
        self.pokemon = self.model_data["pokemon"]
    
    def find_pokemon(self, name: str) -> Optional[pd.Series]:
        """
        Zoek een Pokemon in de database.
        
        Args:
            name: Naam van de Pokemon
        
        Returns:
            Pokemon data als gevonden, anders None
        """
        result = self.pokemon[self.pokemon["Name"].str.lower() == name.lower()]
        
        if result.empty:
            print(f"❌ Pokémon '{name}' niet gevonden!")
            print(f"\n💡 TIP: Probeer een van deze namen:")
            suggestions = self.pokemon[
                self.pokemon["Name"].str.contains(name, case=False, na=False)
            ]["Name"].head(5).tolist()
            if suggestions:
                for s in suggestions:
                    print(f"   - {s}")
            return None
        
        return result.iloc[0]
    
    def predict_battle(self, pokemon1_name: str, pokemon2_name: str) -> Optional[str]:
        """
        Voorspel welke Pokemon zou winnen in een gevecht.
        
        Args:
            pokemon1_name: Naam van eerste Pokemon
            pokemon2_name: Naam van tweede Pokemon
        
        Returns:
            Naam van voorspelde winnaar of None als fout
        """
        # Pokemon zoeken
        p1 = self.find_pokemon(pokemon1_name)
        p2 = self.find_pokemon(pokemon2_name)
        
        if p1 is None or p2 is None:
            return None
        
        # Type advantage berekenen
        type_advantage = calculate_type_advantage(
            str(p1["Type 1"]),
            str(p1["Type 2"]) if pd.notna(p1["Type 2"]) else None,
            str(p2["Type 1"]),
            str(p2["Type 2"]) if pd.notna(p2["Type 2"]) else None
        )
        
        # Features berekenen
        features = pd.DataFrame({
            "HP_diff": [p1["HP"] - p2["HP"]],
            "Attack_diff": [p1["Attack"] - p2["Attack"]],
            "Defense_diff": [p1["Defense"] - p2["Defense"]],
            "Sp_Atk_diff": [p1["Sp. Atk"] - p2["Sp. Atk"]],
            "Sp_Def_diff": [p1["Sp. Def"] - p2["Sp. Def"]],
            "Speed_diff": [p1["Speed"] - p2["Speed"]],
            "Legendary_diff": [int(p1["Legendary"]) - int(p2["Legendary"])],
            "Total_diff": [
                (p1["HP"] + p1["Attack"] + p1["Defense"] + 
                 p1["Sp. Atk"] + p1["Sp. Def"] + p1["Speed"]) -
                (p2["HP"] + p2["Attack"] + p2["Defense"] + 
                 p2["Sp. Atk"] + p2["Sp. Def"] + p2["Speed"])
            ],
            "Type_advantage": [type_advantage]
        })
        
        # Voorspelling
        prediction = self.best_model.predict(features)[0]
        probs = self.best_model.predict_proba(features)[0]
        probability = float(probs[1] if prediction == 1 else probs[0])
        
        # Resultaat printen
        self._print_battle_analysis(p1, p2, prediction, probability, type_advantage)
        
        winner = p1["Name"] if prediction == 1 else p2["Name"]
        return str(winner)
    
    def _print_battle_analysis(self, p1: pd.Series, p2: pd.Series,
                              prediction: int, probability: float,
                              type_advantage: float) -> None:
        """
        Print gedetailleerde battle analyse.
        
        Args:
            p1: Pokemon 1 data
            p2: Pokemon 2 data
            prediction: Voorspelling (1 voor p1, 0 voor p2)
            probability: Confidence van voorspelling
            type_advantage: Type advantage score
        """
        print("\n" + "=" * 60)
        print(f"⚔️  {p1['Name']} VS {p2['Name']} ⚔️")
        print("=" * 60)
        
        # Pokemon 1 stats
        print(f"\n📊 {p1['Name']}:")
        print(f"  Type: {p1['Type 1']}" + 
              (f"/{p1['Type 2']}" if pd.notna(p1['Type 2']) else ""))
        print(f"  HP: {p1['HP']}, Attack: {p1['Attack']}, Defense: {p1['Defense']}")
        print(f"  Sp.Atk: {p1['Sp. Atk']}, Sp.Def: {p1['Sp. Def']}, Speed: {p1['Speed']}")
        total1 = int(p1['HP'] + p1['Attack'] + p1['Defense'] + 
                     p1['Sp. Atk'] + p1['Sp. Def'] + p1['Speed'])
        print(f"  Total: {total1}, Legendary: {'Yes' if p1['Legendary'] else 'No'}")
        
        # Pokemon 2 stats
        print(f"\n📊 {p2['Name']}:")
        print(f"  Type: {p2['Type 1']}" + 
              (f"/{p2['Type 2']}" if pd.notna(p2['Type 2']) else ""))
        print(f"  HP: {p2['HP']}, Attack: {p2['Attack']}, Defense: {p2['Defense']}")
        print(f"  Sp.Atk: {p2['Sp. Atk']}, Sp.Def: {p2['Sp. Def']}, Speed: {p2['Speed']}")
        total2 = int(p2['HP'] + p2['Attack'] + p2['Defense'] + 
                     p2['Sp. Atk'] + p2['Sp. Def'] + p2['Speed'])
        print(f"  Total: {total2}, Legendary: {'Yes' if p2['Legendary'] else 'No'}")
        
        # Type matchup analyse
        print("\n🎯 TYPE MATCHUP ANALYSE:")
        p1_type2 = str(p1['Type 2']) if pd.notna(p1['Type 2']) else None
        p2_type2 = str(p2['Type 2']) if pd.notna(p2['Type 2']) else None
        
        # P1 aanvalt P2
        p1_effectiveness, p1_desc = get_offensive_matchup(
            str(p1['Type 1']), p1_type2, str(p2['Type 1']), p2_type2
        )
        print(f"  {p1['Name']} → {p2['Name']}: {p1_desc} ({p1_effectiveness}x)")
        
        # P2 aanvalt P1
        p2_effectiveness, p2_desc = get_offensive_matchup(
            str(p2['Type 1']), p2_type2, str(p1['Type 1']), p1_type2
        )
        print(f"  {p2['Name']} → {p1['Name']}: {p2_desc} ({p2_effectiveness}x)")
        
        # Algeheel type voordeel
        if type_advantage > 0.5:
            print(f"\n  💪 {p1['Name']} heeft een type voordeel! (+{type_advantage:.2f})")
        elif type_advantage < -0.5:
            print(f"\n  💪 {p2['Name']} heeft een type voordeel! (+{-type_advantage:.2f})")
        else:
            print(f"\n  ⚖️  Type matchup is redelijk evenwichtig")
        
        # Speed vergelijking
        speed_diff = int(p1['Speed'] - p2['Speed'])
        if speed_diff > 0:
            print(f"\n💨 {p1['Name']} is sneller (+{speed_diff} speed)")
        elif speed_diff < 0:
            print(f"\n💨 {p2['Name']} is sneller (+{-speed_diff} speed)")
        
        # Winnaar
        winner = p1['Name'] if prediction == 1 else p2['Name']
        print(f"\n{'🏆' * 3} VOORSPELLING {'🏆' * 3}")
        print(f"\n✨ {winner.upper()} zou winnen!")
        print(f"🎯 Confidence: {probability*100:.1f}%")
        print("=" * 60 + "\n")


def main() -> None:
    """Main functie voor CLI."""
    predictor = BattlePredictor()
    
    # Verschillende gebruiksmogelijkheden
    if len(sys.argv) == 3:
        # Command line argumenten
        pokemon1 = sys.argv[1]
        pokemon2 = sys.argv[2]
        predictor.predict_battle(pokemon1, pokemon2)
    
    elif len(sys.argv) == 1:
        # Interactieve mode
        print("⚔️  Pokémon Battle Predictor ⚔️")
        print("=" * 60)
        print(f"Database: {len(predictor.pokemon)} pokémon beschikbaar\n")
        
        while True:
            print("\n" + "-" * 60)
            pokemon1 = input("🔴 Eerste pokémon (of 'quit' om te stoppen): ").strip()
            if pokemon1.lower() in ['quit', 'exit', 'stop', 'q']:
                print("\n👋 Bedankt voor het gebruiken van Battle Predictor!")
                break
            
            pokemon2 = input("🔵 Tweede pokémon: ").strip()
            if pokemon2.lower() in ['quit', 'exit', 'stop', 'q']:
                print("\n👋 Bedankt voor het gebruiken van Battle Predictor!")
                break
            
            predictor.predict_battle(pokemon1, pokemon2)
    
    else:
        print("❌ Gebruik: python predict.py [pokemon1] [pokemon2]")
        print("\nVoorbeelden:")
        print("  python predict.py Pikachu Charizard")
        print("  python predict.py   (voor interactieve mode)")


if __name__ == "__main__":
    main()
