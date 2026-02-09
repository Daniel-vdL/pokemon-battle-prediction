"""
Pokemon Battle Predictor - Quick Validation Test
Simpele test voor snelle verificatie van model performance.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'types'))

from typing import Dict
import pandas as pd
import pickle

from pokemon_types import calculate_type_advantage


def quick_test() -> None:
    """Voer een snelle test uit op bekende matchups."""
    
    print("⚡ QUICK MODEL VALIDATION TEST")
    print("=" * 60)
    
    # Load model
    with open("pokemon_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data["model"]
    pokemon = pd.read_csv("datasets/pokemon.csv")
    
    # Bekende test cases met verwachte uitkomsten
    test_cases = [
        {
            "p1": "Mewtwo",
            "p2": "Pidgey", 
            "expected": "Mewtwo",
            "min_confidence": 0.95,
            "reason": "Legendary vs weak Pokemon"
        },
        {
            "p1": "Pikachu",
            "p2": "Squirtle",
            "expected": "Pikachu",
            "min_confidence": 0.80,
            "reason": "Electric beats Water (type advantage)"
        },
        {
            "p1": "Onix",
            "p2": "Charmander",
            "expected": "Onix",
            "min_confidence": 0.70,
            "reason": "Rock vs Fire (type + defense)"
        },
        {
            "p1": "Magikarp",
            "p2": "Gyarados",
            "expected": "Gyarados",
            "min_confidence": 0.90,
            "reason": "Pre-evolution vs evolution"
        },
    ]
    
    passed = 0
    total = len(test_cases)
    
    print(f"\nRunning {total} validation tests...\n")
    
    for i, test in enumerate(test_cases, 1):
        p1_name = test["p1"]
        p2_name = test["p2"]
        expected = test["expected"]
        min_conf = test["min_confidence"]
        
        # Find Pokemon
        p1 = pokemon[pokemon["Name"] == p1_name].iloc[0]
        p2 = pokemon[pokemon["Name"] == p2_name].iloc[0]
        
        # Calculate features
        type_adv = calculate_type_advantage(
            str(p1["Type 1"]),
            str(p1["Type 2"]) if pd.notna(p1["Type 2"]) else None,
            str(p2["Type 1"]),
            str(p2["Type 2"]) if pd.notna(p2["Type 2"]) else None
        )
        
        features = pd.DataFrame({
            "HP_diff": [p1["HP"] - p2["HP"]],
            "Attack_diff": [p1["Attack"] - p2["Attack"]],
            "Defense_diff": [p1["Defense"] - p2["Defense"]],
            "Sp_Atk_diff": [p1["Sp. Atk"] - p2["Sp. Atk"]],
            "Sp_Def_diff": [p1["Sp. Def"] - p2["Sp. Def"]],
            "Speed_diff": [p1["Speed"] - p2["Speed"]],
            "Legendary_diff": [int(p1["Legendary"]) - int(p2["Legendary"])],
            "Total_diff": [(p1["HP"] + p1["Attack"] + p1["Defense"] + 
                           p1["Sp. Atk"] + p1["Sp. Def"] + p1["Speed"]) -
                          (p2["HP"] + p2["Attack"] + p2["Defense"] + 
                           p2["Sp. Atk"] + p2["Sp. Def"] + p2["Speed"])],
            "Type_advantage": [type_adv]
        })
        
        # Predict
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        confidence = float(proba[1] if pred == 1 else proba[0])
        winner = p1_name if pred == 1 else p2_name
        
        # Check result
        correct_winner = winner == expected
        correct_confidence = confidence >= min_conf
        test_passed = correct_winner and correct_confidence
        
        if test_passed:
            passed += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        print(f"Test {i}: {p1_name} vs {p2_name}")
        print(f"  Reason: {test['reason']}")
        print(f"  Expected: {expected} (min {min_conf*100:.0f}% conf)")
        print(f"  Got: {winner} ({confidence*100:.1f}% conf)")
        print(f"  {status}")
        
        if not correct_winner:
            print(f"  ⚠️  Wrong winner predicted!")
        if not correct_confidence:
            print(f"  ⚠️  Confidence too low: {confidence*100:.1f}% < {min_conf*100:.0f}%")
        print()
    
    # Summary
    print("=" * 60)
    print(f"\n📊 RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Model is working correctly.")
    elif passed >= total * 0.75:
        print("✓ Most tests passed. Model is performing well.")
    else:
        print("⚠️  Multiple tests failed. Model may need retraining.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    quick_test()
