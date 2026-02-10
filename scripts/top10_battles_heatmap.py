import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Zet pad naar projectroot (een level omhoog van scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# Laad data
combats = pd.read_csv("datasets/combats.csv")
pokemon = pd.read_csv("datasets/pokemon.csv")

# Stat-kolommen
stat_columns = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

# Tel wins per Pokemon
wins = pd.concat([
    combats[combats["Winner"] == combats["First_pokemon"]][["First_pokemon"]].rename(columns={"First_pokemon": "pokemon"}),
    combats[combats["Winner"] == combats["Second_pokemon"]][["Second_pokemon"]].rename(columns={"Second_pokemon": "pokemon"})
])

wins_count = wins["pokemon"].value_counts().rename("Wins")

# Selecteer top 10 op wins
top_10_wins = wins_count.head(10).reset_index()
top_10_wins.columns = ["#", "Wins"]

# Merge met pokemon data
top_10 = top_10_wins.merge(pokemon, on="#", how="left")

# Label met naam en aantal wins
top_10["Label"] = top_10["Name"] + " (" + top_10["Wins"].astype(str) + " wins)"

# Heatmap data
heatmap_data = top_10[stat_columns]
heatmap_data.index = top_10["Label"]

# Visualisatie
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))
ax = sns.heatmap(
    heatmap_data.T,
    annot=True,
    fmt="g",
    cmap="YlOrRd",
    cbar_kws={"label": "Stat Value"},
    yticklabels=stat_columns,
    xticklabels=top_10["Label"]
)

ax.set_title("Stats Heatmap - Top 10 Pokemon op Wins", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

os.makedirs("visualizations", exist_ok=True)
plt.savefig("visualizations/top_10_wins_stats_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ Heatmap opgeslagen: visualizations/top_10_wins_stats_heatmap.png")
