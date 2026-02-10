"""
Pokemon Battle Predictor - Simple Desktop GUI
Select two Pokemon and predict the winner using the trained model.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
import threading

import pandas as pd

# Make types/ available for import
sys.path.append(str(Path(__file__).parent.parent / "types"))

from pokemon_types import calculate_type_advantage, get_offensive_matchup
from team_features import (
    build_single_matchup_features,
    build_team_features,
    build_team_type_map,
)

# Import training classes
sys.path.append(str(Path(__file__).parent))
from train_model import PokemonBattlePredictor
from train_team_model import TeamBattlePredictor


class BattlePredictorGUI:
    """Desktop GUI for Pokemon battle prediction."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pokemon Battle Predictor")
        self.root.geometry("720x520")
        self.root.minsize(640, 480)

        self.model = None
        self.pokemon = pd.DataFrame()

        self.team_model = None
        self.teams = pd.DataFrame()
        self.team_features = pd.DataFrame()
        self.team_types: Dict[int, List[tuple[str, str | None]]] = {}
        self.team_feature_columns: List[str] = []

        self._load_model()
        self._load_team_model()
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

    def _load_team_model(self) -> None:
        model_path = Path(__file__).parent.parent / "team_model.pkl"
        if not model_path.exists():
            return

        with open(model_path, "rb") as f:
            model_data: Dict[str, Any] = pickle.load(f)

        self.team_model = model_data["model"]
        self.teams = model_data["teams"].rename(columns={"#": "team_id"})
        self.team_feature_columns = model_data["feature_columns"]
        self.team_features = model_data.get("team_features", pd.DataFrame())

        if self.team_features.empty:
            self.team_features = build_team_features(self.pokemon, self.teams)

        self.team_types = build_team_type_map(self.pokemon, self.teams)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(main, text="Pokemon Battle Predictor", font=("Segoe UI", 16, "bold"))
        title.pack(pady=(0, 12))

        self.pokemon_names = sorted(self.pokemon["Name"].astype(str).tolist())
        self.team_ids = (
            sorted(self.teams["team_id"].astype(int).tolist())
            if not self.teams.empty
            else []
        )

        notebook = ttk.Notebook(main)
        notebook.pack(fill=tk.BOTH, expand=True)

        pokemon_tab = ttk.Frame(notebook)
        team_tab = ttk.Frame(notebook)
        settings_tab = ttk.Frame(notebook)
        notebook.add(pokemon_tab, text="Pokemon")
        notebook.add(team_tab, text="Team")
        notebook.add(settings_tab, text="Settings")

        self._build_pokemon_tab(pokemon_tab)
        self._build_team_tab(team_tab)
        self._build_settings_tab(settings_tab)

    def _build_pokemon_tab(self, parent: ttk.Frame) -> None:
        selector = ttk.Frame(parent)
        selector.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(selector, text="Pokemon 1").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(selector, text="Pokemon 2").grid(row=0, column=2, sticky=tk.W)

        self.pokemon_p1_var = tk.StringVar()
        self.pokemon_p2_var = tk.StringVar()

        self.pokemon_p1_combo = ttk.Combobox(
            selector, textvariable=self.pokemon_p1_var, values=self.pokemon_names, state="normal"
        )
        self.pokemon_p2_combo = ttk.Combobox(
            selector, textvariable=self.pokemon_p2_var, values=self.pokemon_names, state="normal"
        )
        self._attach_search(self.pokemon_p1_combo)
        self._attach_search(self.pokemon_p2_combo)
        self.pokemon_p1_combo.grid(row=1, column=0, sticky=tk.EW)
        self.pokemon_p2_combo.grid(row=1, column=2, sticky=tk.EW)

        clear_p1 = ttk.Button(selector, text="x", width=2, command=lambda: self._clear_var(self.pokemon_p1_var))
        clear_p2 = ttk.Button(selector, text="x", width=2, command=lambda: self._clear_var(self.pokemon_p2_var))
        clear_p1.grid(row=1, column=1, sticky=tk.W, padx=(4, 12))
        clear_p2.grid(row=1, column=3, sticky=tk.W, padx=(4, 0))

        selector.columnconfigure(0, weight=1)
        selector.columnconfigure(2, weight=1)

        buttons = ttk.Frame(parent)
        buttons.pack(fill=tk.X, pady=(8, 8))

        ttk.Button(buttons, text="Predict", command=self._on_predict_pokemon).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Swap", command=self._on_swap_pokemon).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Button(buttons, text="Clear Output", command=self._on_clear_pokemon).pack(side=tk.LEFT)

        self.pokemon_output = tk.Text(parent, height=18, wrap=tk.WORD, state=tk.DISABLED)
        self.pokemon_output.pack(fill=tk.BOTH, expand=True)

        if self.pokemon_names:
            self.pokemon_p1_var.set(self.pokemon_names[0])
            self.pokemon_p2_var.set(
                self.pokemon_names[1] if len(self.pokemon_names) > 1 else self.pokemon_names[0]
            )

    def _build_team_tab(self, parent: ttk.Frame) -> None:
        if self.team_model is None:
            ttk.Label(
                parent,
                text="team_model.pkl not found. Run scripts/train_team_model.py first.",
                foreground="#aa0000",
            ).pack(anchor=tk.W)

        selector = ttk.Frame(parent)
        selector.pack(fill=tk.X, pady=(8, 8))

        ttk.Label(selector, text="Team A (6 slots)").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(selector, text="Team B (6 slots)").grid(row=0, column=1, sticky=tk.W)

        self.team_a_vars: List[tk.StringVar] = []
        self.team_b_vars: List[tk.StringVar] = []
        self.team_a_combos: List[ttk.Combobox] = []
        self.team_b_combos: List[ttk.Combobox] = []

        for idx in range(6):
            a_var = tk.StringVar()
            b_var = tk.StringVar()
            self.team_a_vars.append(a_var)
            self.team_b_vars.append(b_var)

            a_combo = ttk.Combobox(
                selector, textvariable=a_var, values=self.pokemon_names, state="normal"
            )
            b_combo = ttk.Combobox(
                selector, textvariable=b_var, values=self.pokemon_names, state="normal"
            )
            self._attach_search(a_combo)
            self._attach_search(b_combo)
            a_combo.grid(row=idx + 1, column=0, sticky=tk.EW, pady=2)
            b_combo.grid(row=idx + 1, column=2, sticky=tk.EW, pady=2)

            clear_a = ttk.Button(
                selector, text="x", width=2, command=lambda v=a_var: self._clear_var(v)
            )
            clear_b = ttk.Button(
                selector, text="x", width=2, command=lambda v=b_var: self._clear_var(v)
            )
            clear_a.grid(row=idx + 1, column=1, sticky=tk.W, padx=(4, 12), pady=2)
            clear_b.grid(row=idx + 1, column=3, sticky=tk.W, padx=(4, 0), pady=2)

            self.team_a_combos.append(a_combo)
            self.team_b_combos.append(b_combo)

        selector.columnconfigure(0, weight=1)
        selector.columnconfigure(2, weight=1)

        buttons = ttk.Frame(parent)
        buttons.pack(fill=tk.X, pady=(0, 8))

        self.team_predict_btn = ttk.Button(buttons, text="Predict", command=self._on_predict_team)
        self.team_predict_btn.pack(side=tk.LEFT)
        ttk.Button(buttons, text="Swap", command=self._on_swap_team).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Button(buttons, text="Clear Output", command=self._on_clear_team).pack(side=tk.LEFT)

        self.team_output = tk.Text(parent, height=16, wrap=tk.WORD, state=tk.DISABLED)
        self.team_output.pack(fill=tk.BOTH, expand=True)

        if self.team_model is None:
            self.team_predict_btn.configure(state=tk.DISABLED)
            for combo in self.team_a_combos + self.team_b_combos:
                combo.configure(state="disabled")

        if self.pokemon_names:
            for idx in range(6):
                self.team_a_vars[idx].set(self.pokemon_names[idx % len(self.pokemon_names)])
                self.team_b_vars[idx].set(
                    self.pokemon_names[(idx + 1) % len(self.pokemon_names)]
                )

    def _build_settings_tab(self, parent: ttk.Frame) -> None:
        """Build the settings tab for training models."""
        container = ttk.Frame(parent, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            container,
            text="Model Training",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor=tk.W, pady=(0, 8))

        # Pokemon Model Section
        pokemon_frame = ttk.LabelFrame(container, text="Pokemon Battle Model", padding=12)
        pokemon_frame.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(
            pokemon_frame,
            text="Train the AI model for 1v1 Pokemon battles"
        ).pack(anchor=tk.W, pady=(0, 8))

        pokemon_btn_frame = ttk.Frame(pokemon_frame)
        pokemon_btn_frame.pack(fill=tk.X)

        self.train_pokemon_btn = ttk.Button(
            pokemon_btn_frame,
            text="Train Pokemon Model",
            command=self._on_train_pokemon_model
        )
        self.train_pokemon_btn.pack(side=tk.LEFT)

        self.pokemon_training_status = ttk.Label(pokemon_btn_frame, text="")
        self.pokemon_training_status.pack(side=tk.LEFT, padx=(12, 0))

        # Team Model Section
        team_frame = ttk.LabelFrame(container, text="Team Battle Model", padding=12)
        team_frame.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(
            team_frame,
            text="Train the AI model for 6v6 team battles"
        ).pack(anchor=tk.W, pady=(0, 8))

        team_btn_frame = ttk.Frame(team_frame)
        team_btn_frame.pack(fill=tk.X)

        self.train_team_btn = ttk.Button(
            team_btn_frame,
            text="Train Team Model",
            command=self._on_train_team_model
        )
        self.train_team_btn.pack(side=tk.LEFT)

        self.team_training_status = ttk.Label(team_btn_frame, text="")
        self.team_training_status.pack(side=tk.LEFT, padx=(12, 0))

        # Output log
        ttk.Label(
            container,
            text="Training Log",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor=tk.W, pady=(8, 4))

        log_frame = ttk.Frame(container)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.training_log = tk.Text(log_frame, height=15, wrap=tk.WORD, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(log_frame, command=self.training_log.yview)
        self.training_log.configure(yscrollcommand=scrollbar.set)
        
        self.training_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(
            container,
            text="Clear Log",
            command=lambda: self._set_output(self.training_log, "")
        ).pack(anchor=tk.W, pady=(8, 0))

    def _attach_search(self, combo: ttk.Combobox) -> None:
        combo.bind("<KeyRelease>", lambda event, c=combo: self._schedule_filter(c, event))

    def _schedule_filter(self, combo: ttk.Combobox, event: tk.Event) -> None:
        timer_id = getattr(combo, "_filter_after_id", None)
        if timer_id:
            combo.after_cancel(timer_id)
        combo._filter_after_id = combo.after(120, lambda: self._filter_combobox(combo))

    def _filter_combobox(self, combo: ttk.Combobox) -> None:
        query = combo.get().strip().lower()
        if not query:
            values = self.pokemon_names
        else:
            values = [name for name in self.pokemon_names if query in name.lower()]
        combo.configure(values=values)

    def _clear_var(self, var: tk.StringVar) -> None:
        var.set("")


    def _on_swap_pokemon(self) -> None:
        p1 = self.pokemon_p1_var.get()
        p2 = self.pokemon_p2_var.get()
        self.pokemon_p1_var.set(p2)
        self.pokemon_p2_var.set(p1)

    def _on_clear_pokemon(self) -> None:
        self._set_output(self.pokemon_output, "")

    def _on_predict_pokemon(self) -> None:
        p1_name = self.pokemon_p1_var.get().strip()
        p2_name = self.pokemon_p2_var.get().strip()

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
        self._set_output(self.pokemon_output, output)

    def _on_swap_team(self) -> None:
        for idx in range(6):
            a_val = self.team_a_vars[idx].get()
            b_val = self.team_b_vars[idx].get()
            self.team_a_vars[idx].set(b_val)
            self.team_b_vars[idx].set(a_val)

    def _on_clear_team(self) -> None:
        self._set_output(self.team_output, "")

    def _on_predict_team(self) -> None:
        team_a_names = [var.get().strip() for var in self.team_a_vars]
        team_b_names = [var.get().strip() for var in self.team_b_vars]

        if any(not name for name in team_a_names + team_b_names):
            messagebox.showwarning("Missing selection", "Please fill all 6 slots for each team.")
            return

        team_a_ids = self._names_to_ids(team_a_names)
        team_b_ids = self._names_to_ids(team_b_names)
        if team_a_ids is None or team_b_ids is None:
            messagebox.showwarning("Not found", "One or more Pokemon could not be found.")
            return

        output = self._predict_team_and_format(team_a_ids, team_b_ids, team_a_names, team_b_names)
        self._set_output(self.team_output, output)

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

    def _predict_team_and_format(
        self,
        team_a_ids: List[int],
        team_b_ids: List[int],
        team_a_names: List[str],
        team_b_names: List[str],
    ) -> str:
        temp_teams = pd.DataFrame(
            [
                [1] + team_a_ids,
                [2] + team_b_ids,
            ],
            columns=["team_id", "0", "1", "2", "3", "4", "5"],
        )

        temp_features = build_team_features(self.pokemon, temp_teams)
        temp_types = build_team_type_map(self.pokemon, temp_teams)

        features = build_single_matchup_features(
            1,
            2,
            temp_features,
            temp_types,
            self.team_feature_columns,
        )

        prediction = self.team_model.predict(features)[0]
        probs = self.team_model.predict_proba(features)[0]
        probability = float(probs[1] if prediction == 1 else probs[0])

        lines = []
        lines.append("Team Battle Prediction")
        lines.append("=" * 60)
        lines.append("Custom Team A vs Custom Team B")
        lines.append("")

        lines.append("Team A")
        for name in team_a_names:
            lines.append(f"  - {name}")
        lines.append("")

        lines.append("Team B")
        for name in team_b_names:
            lines.append(f"  - {name}")
        lines.append("")

        winner = "Team A" if prediction == 1 else "Team B"
        lines.append("Prediction")
        lines.append(f"  Winner: {winner}")
        lines.append(f"  Win probability: {probability * 100:.1f}%")

        return "\n".join(lines)

    def _set_output(self, widget: tk.Text, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED)

    def _get_team_members(self, team_id: int) -> pd.DataFrame:
        row = self.teams[self.teams["team_id"] == team_id]
        if row.empty:
            return pd.DataFrame()
        members = row.iloc[0][["0", "1", "2", "3", "4", "5"]].astype(int).tolist()
        return self.pokemon[self.pokemon["#"].isin(members)]

    def _names_to_ids(self, names: List[str]) -> List[int] | None:
        lookup = {
            str(name).lower(): int(pid)
            for pid, name in self.pokemon[["#", "Name"]].itertuples(index=False)
        }
        ids: List[int] = []
        for name in names:
            pid = lookup.get(name.lower())
            if pid is None:
                return None
            ids.append(pid)
        return ids

    def _on_train_pokemon_model(self) -> None:
        """Train the Pokemon battle model in a background thread."""
        if messagebox.askyesno(
            "Train Model",
            "This will train a new Pokemon battle model. This may take a few minutes.\n\nContinue?"
        ):
            self.train_pokemon_btn.configure(state=tk.DISABLED)
            self.pokemon_training_status.configure(text="Training...", foreground="#0066cc")
            self._log_training("Starting Pokemon model training...")
            
            # Run training in background thread
            thread = threading.Thread(target=self._train_pokemon_model_thread, daemon=True)
            thread.start()

    def _train_pokemon_model_thread(self) -> None:
        """Background thread for training Pokemon model."""
        try:
            base_path = Path(__file__).parent.parent
            
            # Create predictor and train
            predictor = PokemonBattlePredictor(
                pokemon_csv=str(base_path / "datasets" / "pokemon.csv"),
                combats_csv=str(base_path / "datasets" / "combats.csv")
            )
            
            self._log_training("Loading data...")
            predictor.load_data()
            
            self._log_training("Preparing features...")
            predictor.prepare_features()
            
            self._log_training("Creating train/test split...")
            predictor.prepare_train_test_split()
            
            self._log_training("Training models (this may take a moment)...")
            results = predictor.train_models()
            
            self._log_training("Selecting best model...")
            predictor.select_best_model(results)
            
            self._log_training("Saving model...")
            predictor.save_model(str(base_path / "pokemon_model.pkl"))
            
            # Log results
            best_acc = results[predictor.best_model_name]["accuracy"]
            self._log_training(f"\n✅ Training complete!")
            self._log_training(f"Best model: {predictor.best_model_name}")
            self._log_training(f"Test accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
            
            # Reload the model in the GUI
            self.root.after(0, self._reload_pokemon_model)
            self.root.after(0, lambda: self.pokemon_training_status.configure(
                text="Training complete!", foreground="#008800"
            ))
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            self._log_training(error_msg)
            self.root.after(0, lambda: self.pokemon_training_status.configure(
                text="Training failed!", foreground="#cc0000"
            ))
        finally:
            self.root.after(0, lambda: self.train_pokemon_btn.configure(state=tk.NORMAL))

    def _on_train_team_model(self) -> None:
        """Train the team battle model in a background thread."""
        if messagebox.askyesno(
            "Train Model",
            "This will train a new team battle model. This may take a few minutes.\n\nContinue?"
        ):
            self.train_team_btn.configure(state=tk.DISABLED)
            self.team_training_status.configure(text="Training...", foreground="#0066cc")
            self._log_training("Starting team model training...")
            
            # Run training in background thread
            thread = threading.Thread(target=self._train_team_model_thread, daemon=True)
            thread.start()

    def _train_team_model_thread(self) -> None:
        """Background thread for training team model."""
        try:
            base_path = Path(__file__).parent.parent
            
            # Create predictor and train
            predictor = TeamBattlePredictor(
                pokemon_csv=str(base_path / "datasets" / "pokemon.csv"),
                teams_csv=str(base_path / "datasets" / "pokemon_id_each_team.csv"),
                battles_csv=str(base_path / "datasets" / "team_combat.csv")
            )
            
            self._log_training("Loading team data...")
            predictor.load_data()
            
            self._log_training("Building team features...")
            predictor.prepare_features()
            
            self._log_training("Training models (this may take a moment)...")
            results = predictor.train_models()
            
            self._log_training("Selecting best model...")
            predictor.select_best_model(results)
            
            self._log_training("Saving model...")
            predictor.save_model(str(base_path / "team_model.pkl"))
            
            # Log results
            best_acc = results[predictor.best_model_name]["accuracy"]
            self._log_training(f"\n✅ Training complete!")
            self._log_training(f"Best model: {predictor.best_model_name}")
            self._log_training(f"Test accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
            
            # Reload the model in the GUI
            self.root.after(0, self._reload_team_model)
            self.root.after(0, lambda: self.team_training_status.configure(
                text="Training complete!", foreground="#008800"
            ))
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            self._log_training(error_msg)
            self.root.after(0, lambda: self.team_training_status.configure(
                text="Training failed!", foreground="#cc0000"
            ))
        finally:
            self.root.after(0, lambda: self.train_team_btn.configure(state=tk.NORMAL))

    def _reload_pokemon_model(self) -> None:
        """Reload the Pokemon model after training."""
        try:
            self._load_model()
            messagebox.showinfo(
                "Model Updated",
                "Pokemon battle model has been updated successfully!"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reload model: {str(e)}")

    def _reload_team_model(self) -> None:
        """Reload the team model after training."""
        try:
            self._load_team_model()
            
            # Re-enable team tab if it was previously disabled
            if self.team_model is not None:
                self.team_predict_btn.configure(state=tk.NORMAL)
                for combo in self.team_a_combos + self.team_b_combos:
                    combo.configure(state="normal")
            
            messagebox.showinfo(
                "Model Updated",
                "Team battle model has been updated successfully!"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reload model: {str(e)}")

    def _log_training(self, message: str) -> None:
        """Append a message to the training log (thread-safe)."""
        def append():
            self.training_log.configure(state=tk.NORMAL)
            self.training_log.insert(tk.END, message + "\n")
            self.training_log.see(tk.END)
            self.training_log.configure(state=tk.DISABLED)
        
        # Schedule on main thread
        self.root.after(0, append)


def main() -> None:
    root = tk.Tk()
    app = BattlePredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
