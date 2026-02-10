"""
Pokemon Visualisatie Script
Dit script maakt grafieken van Pokemon data om patterns te ontdekken.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style voor mooiere grafieken
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. LAAD DE DATA
print("📊 Data laden...")
pokemon = pd.read_csv("datasets/pokemon.csv")

# Laten we eerst zien wat we hebben
print(f"\nTotal Pokemon in dataset: {len(pokemon)}")
print(f"Kolommen: {pokemon.columns.tolist()}")

# 2. BEREKEN TOTALE STATS
# Dit zijn de stat-kolommen (alles behalve Name, Type, Generation, Legendary)
stat_columns = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

# Voeg een kolom toe met de totale stats
pokemon['Total Stats'] = pokemon[stat_columns].sum(axis=1)

# 3. VIND DE TOP 10 STERKSTE
top_10 = pokemon.nlargest(10, 'Total Stats')

print("\n🏆 TOP 10 STERKSTE POKEMON:")
print(top_10[['Name', 'Total Stats', 'Type 1', 'Legendary']])

# 4. MAAK VISUALISATIE 1: Bar chart van top 10
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Maak een bar chart
colors = ['gold' if legendary else 'steelblue' for legendary in top_10['Legendary']]
bars = ax.bar(range(len(top_10)), top_10['Total Stats'], color=colors, edgecolor='black', linewidth=1.5)

# Voeg labels toe
ax.set_xticks(range(len(top_10)))
ax.set_xticklabels(top_10['Name'], rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Total Stats', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Sterkste Pokemon (Totale Stats)', fontsize=14, fontweight='bold')

# Voeg waarden boven de bars toe
for i, (bar, value) in enumerate(zip(bars, top_10['Total Stats'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
            f'{int(value)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Voeg legend toe (goud = legendary, blauw = normal)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='gold', edgecolor='black', label='Legendary'),
                   Patch(facecolor='steelblue', edgecolor='black', label='Normal')]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('visualizations/top_10_pokemon.png', dpi=300, bbox_inches='tight')
print("\n✅ Graafiek opgeslagen: visualizations/top_10_pokemon.png")
plt.show()

# 5. MAAK VISUALISATIE 2: Heatmap van hun stats
fig, ax = plt.subplots(figsize=(10, 6))

# Haal alleen de stats op voor top 10
top_10_stats = top_10[stat_columns]
sns.heatmap(top_10_stats.T, annot=True, fmt='g', cmap='YlOrRd', 
            cbar_kws={'label': 'Stat Value'}, ax=ax,
            xticklabels=top_10['Name'], yticklabels=stat_columns)

ax.set_title('Stats Heatmap - Top 10 Sterkste Pokemon', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('visualizations/top_10_stats_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Heatmap opgeslagen: visualizations/top_10_stats_heatmap.png")
plt.show()

print("\n🎉 Visualisaties klaar!")
