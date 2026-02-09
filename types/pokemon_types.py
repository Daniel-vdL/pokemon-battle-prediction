"""
Pokemon Type System Module
Dit module bevat alle Pokemon type effectiveness data en hulpfuncties.
"""
from typing import Dict, List, Tuple, Optional
from enum import Enum


class PokemonType(Enum):
    """Enum voor alle Pokemon types"""
    NORMAL = "Normal"
    FIRE = "Fire"
    WATER = "Water"
    ELECTRIC = "Electric"
    GRASS = "Grass"
    ICE = "Ice"
    FIGHTING = "Fighting"
    POISON = "Poison"
    GROUND = "Ground"
    FLYING = "Flying"
    PSYCHIC = "Psychic"
    BUG = "Bug"
    ROCK = "Rock"
    GHOST = "Ghost"
    DRAGON = "Dragon"
    DARK = "Dark"
    STEEL = "Steel"
    FAIRY = "Fairy"


# Type effectiveness matrix: attackertype -> {defendertype: multiplier}
# 2.0 = super effective, 0.5 = not very effective, 0.0 = no effect
TYPE_EFFECTIVENESS: Dict[str, Dict[str, float]] = {
    "Normal": {
        "Rock": 0.5, "Ghost": 0.0, "Steel": 0.5
    },
    "Fire": {
        "Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 2.0, "Bug": 2.0,
        "Rock": 0.5, "Dragon": 0.5, "Steel": 2.0
    },
    "Water": {
        "Fire": 2.0, "Water": 0.5, "Grass": 0.5, "Ground": 2.0, "Rock": 2.0,
        "Dragon": 0.5
    },
    "Electric": {
        "Water": 2.0, "Electric": 0.5, "Grass": 0.5, "Ground": 0.0,
        "Flying": 2.0, "Dragon": 0.5
    },
    "Grass": {
        "Fire": 0.5, "Water": 2.0, "Grass": 0.5, "Poison": 0.5, "Ground": 2.0,
        "Flying": 0.5, "Bug": 0.5, "Rock": 2.0, "Dragon": 0.5, "Steel": 0.5
    },
    "Ice": {
        "Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 0.5, "Ground": 2.0,
        "Flying": 2.0, "Dragon": 2.0, "Steel": 0.5
    },
    "Fighting": {
        "Normal": 2.0, "Ice": 2.0, "Poison": 0.5, "Flying": 0.5, "Psychic": 0.5,
        "Bug": 0.5, "Rock": 2.0, "Ghost": 0.0, "Dark": 2.0, "Steel": 2.0,
        "Fairy": 0.5
    },
    "Poison": {
        "Grass": 2.0, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5, "Ghost": 0.5,
        "Steel": 0.0, "Fairy": 2.0
    },
    "Ground": {
        "Fire": 2.0, "Electric": 2.0, "Grass": 0.5, "Poison": 2.0, "Flying": 0.0,
        "Bug": 0.5, "Rock": 2.0, "Steel": 2.0
    },
    "Flying": {
        "Electric": 0.5, "Grass": 2.0, "Fighting": 2.0, "Bug": 2.0, "Rock": 0.5,
        "Steel": 0.5
    },
    "Psychic": {
        "Fighting": 2.0, "Poison": 2.0, "Psychic": 0.5, "Dark": 0.0, "Steel": 0.5
    },
    "Bug": {
        "Fire": 0.5, "Grass": 2.0, "Fighting": 0.5, "Poison": 0.5, "Flying": 0.5,
        "Psychic": 2.0, "Ghost": 0.5, "Dark": 2.0, "Steel": 0.5, "Fairy": 0.5
    },
    "Rock": {
        "Fire": 2.0, "Ice": 2.0, "Fighting": 0.5, "Ground": 0.5, "Flying": 2.0,
        "Bug": 2.0, "Steel": 0.5
    },
    "Ghost": {
        "Normal": 0.0, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5
    },
    "Dragon": {
        "Dragon": 2.0, "Steel": 0.5, "Fairy": 0.0
    },
    "Dark": {
        "Fighting": 0.5, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5, "Fairy": 0.5
    },
    "Steel": {
        "Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2.0, "Rock": 2.0,
        "Steel": 0.5, "Fairy": 2.0
    },
    "Fairy": {
        "Fire": 0.5, "Fighting": 2.0, "Poison": 0.5, "Dragon": 2.0, "Dark": 2.0,
        "Steel": 0.5
    }
}


def get_type_effectiveness(attacker_type: str, defender_type1: str, 
                          defender_type2: Optional[str] = None) -> float:
    """
    Bereken de type effectiveness van een aanval.
    
    Args:
        attacker_type: Het type van de aanvaller
        defender_type1: Het primaire type van de verdediger
        defender_type2: Het secundaire type van de verdediger (optioneel)
    
    Returns:
        Type effectiveness multiplier (0, 0.25, 0.5, 1, 2, of 4)
    """
    multiplier = 1.0
    
    # Check effectiveness tegen type 1
    if attacker_type in TYPE_EFFECTIVENESS:
        if defender_type1 in TYPE_EFFECTIVENESS[attacker_type]:
            multiplier *= TYPE_EFFECTIVENESS[attacker_type][defender_type1]
    
    # Check effectiveness tegen type 2 (als aanwezig)
    if defender_type2:
        if attacker_type in TYPE_EFFECTIVENESS:
            if defender_type2 in TYPE_EFFECTIVENESS[attacker_type]:
                multiplier *= TYPE_EFFECTIVENESS[attacker_type][defender_type2]
    
    return multiplier


def get_offensive_matchup(attacker_type1: str, attacker_type2: Optional[str],
                         defender_type1: str, defender_type2: Optional[str]) -> Tuple[float, str]:
    """
    Bereken de algemene offensive type matchup tussen twee Pokemon.
    
    Args:
        attacker_type1: Primair type van aanvaller
        attacker_type2: Secundair type van aanvaller (optioneel)
        defender_type1: Primair type van verdediger
        defender_type2: Secundair type van verdediger (optioneel)
    
    Returns:
        Tuple van (beste_effectiveness, beschrijving)
    """
    effectiveness_type1 = get_type_effectiveness(attacker_type1, defender_type1, defender_type2)
    
    if attacker_type2:
        effectiveness_type2 = get_type_effectiveness(attacker_type2, defender_type1, defender_type2)
        best_effectiveness = max(effectiveness_type1, effectiveness_type2)
    else:
        best_effectiveness = effectiveness_type1
    
    # Beschrijving genereren
    if best_effectiveness >= 2.0:
        description = "Super Effective"
    elif best_effectiveness > 1.0:
        description = "Effective"
    elif best_effectiveness == 1.0:
        description = "Normal"
    elif best_effectiveness > 0.0:
        description = "Not Very Effective"
    else:
        description = "No Effect"
    
    return best_effectiveness, description


def calculate_type_advantage(pokemon1_type1: str, pokemon1_type2: Optional[str],
                            pokemon2_type1: str, pokemon2_type2: Optional[str]) -> float:
    """
    Bereken het algehele type voordeel tussen twee Pokemon.
    
    Args:
        pokemon1_type1: Primair type Pokemon 1
        pokemon1_type2: Secundair type Pokemon 1
        pokemon2_type1: Primair type Pokemon 2
        pokemon2_type2: Secundair type Pokemon 2
    
    Returns:
        Type advantage score (positief = voordeel voor Pokemon 1)
    """
    # Pokemon 1 aanvalt Pokemon 2
    p1_offense, _ = get_offensive_matchup(pokemon1_type1, pokemon1_type2, 
                                         pokemon2_type1, pokemon2_type2)
    
    # Pokemon 2 aanvalt Pokemon 1
    p2_offense, _ = get_offensive_matchup(pokemon2_type1, pokemon2_type2,
                                         pokemon1_type1, pokemon1_type2)
    
    # Verschil berekenen (positief = voordeel voor Pokemon 1)
    return p1_offense - p2_offense


def get_weaknesses(pokemon_type1: str, pokemon_type2: Optional[str] = None) -> List[Tuple[str, float]]:
    """
    Krijg alle weaknesses van een Pokemon type combinatie.
    
    Args:
        pokemon_type1: Primair type
        pokemon_type2: Secundair type (optioneel)
    
    Returns:
        List van (type, effectiveness) tuples waar effectiveness > 1.0
    """
    weaknesses: List[Tuple[str, float]] = []
    
    for attacker_type in TYPE_EFFECTIVENESS.keys():
        effectiveness = get_type_effectiveness(attacker_type, pokemon_type1, pokemon_type2)
        if effectiveness > 1.0:
            weaknesses.append((attacker_type, effectiveness))
    
    # Sorteer op effectiveness (hoogste eerst)
    weaknesses.sort(key=lambda x: x[1], reverse=True)
    return weaknesses


def get_resistances(pokemon_type1: str, pokemon_type2: Optional[str] = None) -> List[Tuple[str, float]]:
    """
    Krijg alle resistances van een Pokemon type combinatie.
    
    Args:
        pokemon_type1: Primair type
        pokemon_type2: Secundair type (optioneel)
    
    Returns:
        List van (type, effectiveness) tuples waar effectiveness < 1.0
    """
    resistances: List[Tuple[str, float]] = []
    
    for attacker_type in TYPE_EFFECTIVENESS.keys():
        effectiveness = get_type_effectiveness(attacker_type, pokemon_type1, pokemon_type2)
        if effectiveness < 1.0:
            resistances.append((attacker_type, effectiveness))
    
    # Sorteer op effectiveness (laagste eerst)
    resistances.sort(key=lambda x: x[1])
    return resistances


def format_type_matchup_info(type1: str, type2: Optional[str] = None) -> str:
    """
    Formatteer een leesbare string met type matchup informatie.
    
    Args:
        type1: Primair type
        type2: Secundair type (optioneel)
    
    Returns:
        Geformatteerde string met weaknesses en resistances
    """
    output = []
    
    weaknesses = get_weaknesses(type1, type2)
    resistances = get_resistances(type1, type2)
    
    if weaknesses:
        output.append("  Weaknesses:")
        for weak_type, effectiveness in weaknesses:
            multiplier_str = f"{effectiveness}x" if effectiveness != 2.0 else "2x"
            output.append(f"    - {weak_type} ({multiplier_str})")
    
    if resistances:
        output.append("  Resistances:")
        for resist_type, effectiveness in resistances:
            if effectiveness == 0.0:
                output.append(f"    - {resist_type} (Immune)")
            else:
                output.append(f"    - {resist_type} ({effectiveness}x)")
    
    return "\n".join(output)
