"""
Pokemon Battle Predictor - Random Battle Generator
Genereer random battles en sla resultaten op voor analyse en visualisatie.

Usage:
    python generate_battles.py                # Genereer 10 random battles
    python generate_battles.py --count 50     # Genereer 50 battles
    python generate_battles.py --legendary    # Alleen legendary Pokemon
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'types'))

from typing import List, Dict, Any
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

from pokemon_types import calculate_type_advantage, get_offensive_matchup


class BattleGenerator:
    """Class voor het genereren van random Pokemon battles."""
    
    def __init__(self, model_path: str = "pokemon_model.pkl") -> None:
        """
        Initialiseer de battle generator.
        
        Args:
            model_path: Path naar opgeslagen model
        """
        print("📦 Model en data laden...")
        with open(model_path, "rb") as f:
            self.model_data = pickle.load(f)
        
        self.model = self.model_data["model"]
        self.pokemon = pd.read_csv("datasets/pokemon.csv")
        print(f"✓ Loaded {len(self.pokemon)} Pokemon\n")
    
    def generate_random_battles(self, count: int = 10, 
                               legendary_only: bool = False,
                               no_legendary: bool = False) -> List[Dict[str, Any]]:
        """
        Genereer random battles en voorspel de uitkomsten.
        
        Args:
            count: Aantal battles om te genereren
            legendary_only: Alleen legendary Pokemon gebruiken
            no_legendary: Geen legendary Pokemon gebruiken
        
        Returns:
            List van battle result dictionaries
        """
        battles: List[Dict[str, Any]] = []
        
        # Filter Pokemon indien nodig
        available_pokemon = self.pokemon.copy()
        if legendary_only:
            available_pokemon = available_pokemon[available_pokemon["Legendary"] == True]
            print(f"🏆 Legendary only mode: {len(available_pokemon)} Pokemon available")
        elif no_legendary:
            available_pokemon = available_pokemon[available_pokemon["Legendary"] == False]
            print(f"⭐ No legendary mode: {len(available_pokemon)} Pokemon available")
        
        if len(available_pokemon) < 2:
            print("❌ Not enough Pokemon available!")
            return []
        
        print(f"🎲 Generating {count} random battles...\n")
        
        for i in range(count):
            # Select 2 random Pokemon
            sample = available_pokemon.sample(n=2)
            p1 = sample.iloc[0]
            p2 = sample.iloc[1]
            
            # Calculate features
            type_advantage = calculate_type_advantage(
                str(p1["Type 1"]),
                str(p1["Type 2"]) if pd.notna(p1["Type 2"]) else None,
                str(p2["Type 1"]),
                str(p2["Type 2"]) if pd.notna(p2["Type 2"]) else None
            )
            
            # Get type matchup details
            p1_type2 = str(p1["Type 2"]) if pd.notna(p1["Type 2"]) else None
            p2_type2 = str(p2["Type 2"]) if pd.notna(p2["Type 2"]) else None
            
            p1_offensive, p1_desc = get_offensive_matchup(
                str(p1["Type 1"]), p1_type2, str(p2["Type 1"]), p2_type2
            )
            p2_offensive, p2_desc = get_offensive_matchup(
                str(p2["Type 1"]), p2_type2, str(p1["Type 1"]), p1_type2
            )
            
            # Calculate total stats
            p1_total = int(p1["HP"] + p1["Attack"] + p1["Defense"] + 
                          p1["Sp. Atk"] + p1["Sp. Def"] + p1["Speed"])
            p2_total = int(p2["HP"] + p2["Attack"] + p2["Defense"] + 
                          p2["Sp. Atk"] + p2["Sp. Def"] + p2["Speed"])
            
            features = pd.DataFrame({
                "HP_diff": [p1["HP"] - p2["HP"]],
                "Attack_diff": [p1["Attack"] - p2["Attack"]],
                "Defense_diff": [p1["Defense"] - p2["Defense"]],
                "Sp_Atk_diff": [p1["Sp. Atk"] - p2["Sp. Atk"]],
                "Sp_Def_diff": [p1["Sp. Def"] - p2["Sp. Def"]],
                "Speed_diff": [p1["Speed"] - p2["Speed"]],
                "Legendary_diff": [int(p1["Legendary"]) - int(p2["Legendary"])],
                "Total_diff": [p1_total - p2_total],
                "Type_advantage": [type_advantage]
            })
            
            # Predict
            prediction = self.model.predict(features)[0]
            proba = self.model.predict_proba(features)[0]
            confidence = float(proba[1]) if prediction == 1 else float(proba[0])
            winner = p1["Name"] if prediction == 1 else p2["Name"]
            loser = p2["Name"] if prediction == 1 else p1["Name"]
            
            # Create battle record
            battle_result = {
                "battle_id": i + 1,
                "pokemon1_name": str(p1["Name"]),
                "pokemon1_type1": str(p1["Type 1"]),
                "pokemon1_type2": str(p1["Type 2"]) if pd.notna(p1["Type 2"]) else None,
                "pokemon1_hp": int(p1["HP"]),
                "pokemon1_attack": int(p1["Attack"]),
                "pokemon1_defense": int(p1["Defense"]),
                "pokemon1_sp_atk": int(p1["Sp. Atk"]),
                "pokemon1_sp_def": int(p1["Sp. Def"]),
                "pokemon1_speed": int(p1["Speed"]),
                "pokemon1_total": p1_total,
                "pokemon1_legendary": bool(p1["Legendary"]),
                "pokemon1_generation": int(p1["Generation"]),
                
                "pokemon2_name": str(p2["Name"]),
                "pokemon2_type1": str(p2["Type 1"]),
                "pokemon2_type2": str(p2["Type 2"]) if pd.notna(p2["Type 2"]) else None,
                "pokemon2_hp": int(p2["HP"]),
                "pokemon2_attack": int(p2["Attack"]),
                "pokemon2_defense": int(p2["Defense"]),
                "pokemon2_sp_atk": int(p2["Sp. Atk"]),
                "pokemon2_sp_def": int(p2["Sp. Def"]),
                "pokemon2_speed": int(p2["Speed"]),
                "pokemon2_total": p2_total,
                "pokemon2_legendary": bool(p2["Legendary"]),
                "pokemon2_generation": int(p2["Generation"]),
                
                "predicted_winner": str(winner),
                "predicted_loser": str(loser),
                "confidence": round(confidence * 100, 2),
                "confidence_raw": round(confidence, 4),
                
                "type_advantage": round(float(type_advantage), 3),
                "pokemon1_offensive_effectiveness": float(p1_offensive),
                "pokemon1_offensive_description": p1_desc,
                "pokemon2_offensive_effectiveness": float(p2_offensive),
                "pokemon2_offensive_description": p2_desc,
                
                "stat_difference": p1_total - p2_total,
                "speed_difference": int(p1["Speed"] - p2["Speed"]),
                "hp_difference": int(p1["HP"] - p2["HP"]),
                "attack_difference": int(p1["Attack"] - p2["Attack"]),
                "defense_difference": int(p1["Defense"] - p2["Defense"]),
            }
            
            battles.append(battle_result)
            
            # Print progress
            print(f"Battle {i+1}: {p1['Name']} vs {p2['Name']} → {winner} wins ({confidence*100:.1f}%)")
        
        return battles
    
    def save_to_json(self, battles: List[Dict[str, Any]], 
                     filename: str = "battles.json") -> None:
        """
        Sla battles op in JSON formaat.
        
        Args:
            battles: List van battle results
            filename: Output filename
        """
        output = {
            "generated_at": datetime.now().isoformat(),
            "total_battles": len(battles),
            "battles": battles
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved to {filename}")
    
    def save_to_csv(self, battles: List[Dict[str, Any]], 
                    filename: str = "battles.csv") -> None:
        """
        Sla battles op in CSV formaat.
        
        Args:
            battles: List van battle results
            filename: Output filename
        """
        df = pd.DataFrame(battles)
        df.to_csv(filename, index=False, encoding="utf-8")
        print(f"✅ Saved to {filename}")
    
    def print_summary(self, battles: List[Dict[str, Any]]) -> None:
        """
        Print een samenvatting van de gegenereerde battles.
        
        Args:
            battles: List van battle results
        """
        if not battles:
            return
        
        df = pd.DataFrame(battles)
        
        print("\n" + "=" * 60)
        print("📊 BATTLE SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal Battles: {len(battles)}")
        print(f"Average Confidence: {df['confidence'].mean():.1f}%")
        print(f"Min Confidence: {df['confidence'].min():.1f}%")
        print(f"Max Confidence: {df['confidence'].max():.1f}%")
        
        # Type advantage distribution
        positive_type_adv = len(df[df['type_advantage'] > 0.5])
        negative_type_adv = len(df[df['type_advantage'] < -0.5])
        neutral = len(df) - positive_type_adv - negative_type_adv
        
        print(f"\nType Advantages:")
        print(f"  Pokemon 1 advantage: {positive_type_adv}")
        print(f"  Neutral: {neutral}")
        print(f"  Pokemon 2 advantage: {negative_type_adv}")
        
        # Legendary stats
        legendary_wins = len(df[df['predicted_winner'].isin(
            df[df['pokemon1_legendary']]['pokemon1_name'].tolist() + 
            df[df['pokemon2_legendary']]['pokemon2_name'].tolist()
        )])
        
        print(f"\nLegendary Pokemon:")
        print(f"  Battles with legendary: {len(df[(df['pokemon1_legendary']) | (df['pokemon2_legendary'])])}")
        
        # Confidence distribution
        high_conf = len(df[df['confidence'] >= 90])
        medium_conf = len(df[(df['confidence'] >= 70) & (df['confidence'] < 90)])
        low_conf = len(df[df['confidence'] < 70])
        
        print(f"\nConfidence Distribution:")
        print(f"  High (≥90%): {high_conf}")
        print(f"  Medium (70-90%): {medium_conf}")
        print(f"  Low (<70%): {low_conf}")
        
        print("\n" + "=" * 60)


def main() -> None:
    """Main functie voor battle generation."""
    # Parse arguments
    count = 10
    legendary_only = False
    no_legendary = False
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg in ["--count", "-c"]:
            if i + 2 < len(sys.argv):
                count = int(sys.argv[i + 2])
        elif arg in ["--legendary", "-l"]:
            legendary_only = True
        elif arg in ["--no-legendary", "-n"]:
            no_legendary = True
        elif arg in ["--help", "-h"]:
            print(__doc__)
            print("\nOptions:")
            print("  --count, -c <n>      Generate <n> battles (default: 10)")
            print("  --legendary, -l      Only use legendary Pokemon")
            print("  --no-legendary, -n   Exclude legendary Pokemon")
            print("  --help, -h           Show this help message")
            print("\nExamples:")
            print("  python generate_battles.py")
            print("  python generate_battles.py --count 50")
            print("  python generate_battles.py --legendary --count 20")
            return
    
    print("🔥 POKEMON BATTLE GENERATOR")
    print("=" * 60)
    
    # Generate battles
    generator = BattleGenerator()
    battles = generator.generate_random_battles(
        count=count,
        legendary_only=legendary_only,
        no_legendary=no_legendary
    )
    
    if battles:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"battles_{timestamp}.json"
        csv_filename = f"battles_{timestamp}.csv"
        
        generator.save_to_json(battles, json_filename)
        generator.save_to_csv(battles, csv_filename)
        
        # Print summary
        generator.print_summary(battles)
        
        print(f"\n💡 Tip: Use these files to create visualizations!")
        print(f"   - {json_filename} (for web apps/detailed analysis)")
        print(f"   - {csv_filename} (for Excel/Python plotting)")


if __name__ == "__main__":
    main()
