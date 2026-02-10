"""
Pokemon Battle Predictor - Simple Desktop GUI
Select two Pokemon and predict the winner using the trained model.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
import pickle
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd

# Make types/ available for import
sys.path.append(str(Path(__file__).parent.parent / "types"))

from pokemon_types import calculate_type_advantage, get_offensive_matchup


class BattlePredictorGUI:
    """Desktop GUI for Pokemon battle prediction."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pokemon Battle Predictor")
        self.root.geometry("720x520")
        self.root.minsize(640, 480)

        self.model = None
        self.pokemon = pd.DataFrame()

        self._load_model()
        self._build_ui()

    def _load_model(self) -> None:
        model_path = Path(__file__).parent.parent / "pokemon_model.pkl"
        if not model_path.exists():
            messagebox.showerror(
                "Model not found",
                "pokemon_model.pkl was not found. Run scripts/train_model.py first."
            )
            self.root.destroy()
            return

        with open(model_path, "rb") as f:
            model_data: Dict[str, Any] = pickle.load(f)

        self.model = model_data["model"]
        self.pokemon = model_data["pokemon"]

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(main, text="Pokemon Battle Predictor", font=("Segoe UI", 16, "bold"))
        title.pack(pady=(0, 12))

        selector = ttk.Frame(main)
        selector.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(selector, text="Pokemon 1").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(selector, text="Pokemon 2").grid(row=0, column=1, sticky=tk.W)

        self.pokemon_names = sorted(self.pokemon["Name"].astype(str).tolist())

        self.p1_var = tk.StringVar(value=self.pokemon_names[0] if self.pokemon_names else "")
        self.p2_var = tk.StringVar(value=self.pokemon_names[1] if len(self.pokemon_names) > 1 else "")

        self.p1_combo = ttk.Combobox(selector, textvariable=self.p1_var, values=self.pokemon_names, state="normal")
        self.p2_combo = ttk.Combobox(selector, textvariable=self.p2_var, values=self.pokemon_names, state="normal")
        self.p1_combo.grid(row=1, column=0, sticky=tk.EW, padx=(0, 8))
        self.p2_combo.grid(row=1, column=1, sticky=tk.EW)

        selector.columnconfigure(0, weight=1)
        selector.columnconfigure(1, weight=1)

        buttons = ttk.Frame(main)
        buttons.pack(fill=tk.X, pady=(8, 8))

        predict_btn = ttk.Button(buttons, text="Predict", command=self._on_predict)
        swap_btn = ttk.Button(buttons, text="Swap", command=self._on_swap)
        clear_btn = ttk.Button(buttons, text="Clear Output", command=self._on_clear)

        predict_btn.pack(side=tk.LEFT)
        swap_btn.pack(side=tk.LEFT, padx=(8, 8))
        clear_btn.pack(side=tk.LEFT)

        self.output = tk.Text(main, height=18, wrap=tk.WORD, state=tk.DISABLED)
        self.output.pack(fill=tk.BOTH, expand=True)

    def _on_swap(self) -> None:
        p1 = self.p1_var.get()
        p2 = self.p2_var.get()
        self.p1_var.set(p2)
        self.p2_var.set(p1)

    def _on_clear(self) -> None:
        self._set_output("")

    def _on_predict(self) -> None:
        p1_name = self.p1_var.get().strip()
        p2_name = self.p2_var.get().strip()

        if not p1_name or not p2_name:
            messagebox.showwarning("Missing selection", "Please select two Pokemon.")
            return

        if p1_name == p2_name:
            messagebox.showwarning("Invalid selection", "Please select two different Pokemon.")
            return

        p1 = self._find_pokemon(p1_name)
        p2 = self._find_pokemon(p2_name)
        if p1 is None or p2 is None:
            messagebox.showwarning("Not found", "One or both Pokemon could not be found.")
            return

        output = self._predict_and_format(p1, p2)
        self._set_output(output)

    def _find_pokemon(self, name: str) -> pd.Series | None:
        result = self.pokemon[self.pokemon["Name"].str.lower() == name.lower()]
        if result.empty:
            return None
        return result.iloc[0]

    def _predict_and_format(self, p1: pd.Series, p2: pd.Series) -> str:
        type_advantage = calculate_type_advantage(
            str(p1["Type 1"]),
            str(p1["Type 2"]) if pd.notna(p1["Type 2"]) else None,
            str(p2["Type 1"]),
            str(p2["Type 2"]) if pd.notna(p2["Type 2"]) else None,
        )

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
            "Type_advantage": [type_advantage],
        })

        prediction = self.model.predict(features)[0]
        probs = self.model.predict_proba(features)[0]
        probability = float(probs[1] if prediction == 1 else probs[0])

        p1_type2 = str(p1["Type 2"]) if pd.notna(p1["Type 2"]) else None
        p2_type2 = str(p2["Type 2"]) if pd.notna(p2["Type 2"]) else None

        p1_eff, p1_desc = get_offensive_matchup(str(p1["Type 1"]), p1_type2, str(p2["Type 1"]), p2_type2)
        p2_eff, p2_desc = get_offensive_matchup(str(p2["Type 1"]), p2_type2, str(p1["Type 1"]), p1_type2)

        winner = p1["Name"] if prediction == 1 else p2["Name"]

        lines = []
        lines.append("Battle Prediction")
        lines.append("=" * 60)
        lines.append(f"{p1['Name']} vs {p2['Name']}")
        lines.append("")

        lines.append("Pokemon 1")
        lines.append(f"  Name: {p1['Name']}")
        lines.append(f"  Type: {p1['Type 1']}" + (f"/{p1['Type 2']}" if pd.notna(p1['Type 2']) else ""))
        lines.append(f"  HP: {p1['HP']}  Attack: {p1['Attack']}  Defense: {p1['Defense']}")
        lines.append(f"  Sp.Atk: {p1['Sp. Atk']}  Sp.Def: {p1['Sp. Def']}  Speed: {p1['Speed']}")
        lines.append("")

        lines.append("Pokemon 2")
        lines.append(f"  Name: {p2['Name']}")
        lines.append(f"  Type: {p2['Type 1']}" + (f"/{p2['Type 2']}" if pd.notna(p2['Type 2']) else ""))
        lines.append(f"  HP: {p2['HP']}  Attack: {p2['Attack']}  Defense: {p2['Defense']}")
        lines.append(f"  Sp.Atk: {p2['Sp. Atk']}  Sp.Def: {p2['Sp. Def']}  Speed: {p2['Speed']}")
        lines.append("")

        lines.append("Type Matchup")
        lines.append(f"  {p1['Name']} -> {p2['Name']}: {p1_desc} ({p1_eff}x)")
        lines.append(f"  {p2['Name']} -> {p1['Name']}: {p2_desc} ({p2_eff}x)")
        if type_advantage > 0.5:
            lines.append(f"  Advantage: {p1['Name']} (+{type_advantage:.2f})")
        elif type_advantage < -0.5:
            lines.append(f"  Advantage: {p2['Name']} (+{-type_advantage:.2f})")
        else:
            lines.append("  Advantage: fairly even")
        lines.append("")

        lines.append("Prediction")
        lines.append(f"  Winner: {str(winner)}")
        lines.append(f"  Confidence: {probability * 100:.1f}%")

        return "\n".join(lines)

    def _set_output(self, text: str) -> None:
        self.output.configure(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        self.output.configure(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    app = BattlePredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
