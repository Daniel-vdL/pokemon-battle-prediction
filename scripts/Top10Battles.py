import pandas as pd
import matplotlib.pyplot as plt
import os

# Zet pad naar projectroot (één level omhoog van scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# Nu laad je de data
combats = pd.read_csv('datasets/combats.csv')
pokemon_data = pd.read_csv('datasets/pokemon.csv')

# Maak een mapping van ID naar Naam
pokemon_names = pokemon_data.set_index('#')['Name'].to_dict()

# Tel alle battles per Pokemon
all_battles = pd.concat([
    combats[['First_pokemon']].rename(columns={'First_pokemon': 'pokemon'}),
    combats[['Second_pokemon']].rename(columns={'Second_pokemon': 'pokemon'})
])

battles_count = all_battles['pokemon'].value_counts()

# Tel wins per Pokemon
wins = pd.concat([
    combats[combats['Winner'] == combats['First_pokemon']][['First_pokemon']].rename(columns={'First_pokemon': 'pokemon'}),
    combats[combats['Winner'] == combats['Second_pokemon']][['Second_pokemon']].rename(columns={'Second_pokemon': 'pokemon'})
])

wins_count = wins['pokemon'].value_counts()

# Combineer in DataFrame
results = pd.DataFrame({
    'Total_Battles': battles_count,
    'Wins': wins_count.fillna(0)
})

# Bereken winpercentage
results['Win_Percentage'] = (results['Wins'] / results['Total_Battles'] * 100).round(2)

# Vervang ID door Pokemon naam
results['Pokemon_Name'] = results.index.map(pokemon_names)
results = results.set_index('Pokemon_Name')

# Filter: minimum 15 battles
top_10 = results[results['Total_Battles'] >= 15].nlargest(10, 'Win_Percentage')

print(top_10)

# Visualisatie
plt.figure(figsize=(10, 6))
plt.scatter(
    top_10["Total_Battles"],
    top_10["Win_Percentage"],
    s=200,
    alpha=0.6,
    c=top_10["Win_Percentage"],
    cmap='RdYlGn'
)

for pokemon in top_10.index:
    plt.text(
        top_10.loc[pokemon, "Total_Battles"],
        top_10.loc[pokemon, "Win_Percentage"],
        str(pokemon),
        fontsize=9,
        ha='left',
        va='bottom'
    )

plt.xlabel("Total Battles", fontsize=12)
plt.ylabel("Win Percentage (%)", fontsize=12)
plt.title("Top 10 Pokemon - Win Percentage vs Total Battles", fontsize=14, fontweight='bold')
plt.colorbar(label='Win %')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/top_10_scatter.png', dpi=300, bbox_inches='tight')
plt.show()