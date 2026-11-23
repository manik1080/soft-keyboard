"""Tkinter presentation layer for the Smart Touch Keyboard.

This module builds widgets and renders model data. It does not implement the
probability model or manage the keyboard's text/calibration workflow.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional

from model import CALIBRATION_PHRASE, KeyboardModel


class KeyboardUI:
    BG = "#0b121a"
    PANEL = "#111e29"
    PANEL_2 = "#0d1822"
    LINE = "#2b4050"
    TEXT = "#ecf4f8"
    MUTED = "#9aabba"
    TEAL = "#62d9c5"
    BLUE = "#9cb8ff"
    ORANGE = "#ffbf69"

    def __init__(self, root: tk.Tk, callbacks: Dict[str, Callable]) -> None:
        self.root = root
        self.callbacks = callbacks

        self.show_gaussians = tk.BooleanVar(value=True)
        self.show_boxes = tk.BooleanVar(value=True)
        self.use_language = tk.BooleanVar(value=True)

        self._configure_window()
        self._build_widgets()

    def _configure_window(self) -> None:
        self.root.title("Keyboard")
        self.root.geometry("1320x850")
        self.root.minsize(940, 720)
        self.root.configure(bg=self.BG)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "TCheckbutton",
            background=self.PANEL,
            foreground=self.TEXT,
            font=("Segoe UI", 9),
            focuscolor=self.PANEL,
        )
        style.map(
            "TCheckbutton",
            background=[("active", self.PANEL)],
            foreground=[("disabled", self.MUTED)],
        )
        style.configure(
            "TButton",
            background="#223746",
            foreground=self.TEXT,
            padding=(9, 5),
            borderwidth=1,
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "TButton",
            background=[("active", "#304d5e"), ("pressed", "#172a37")],
            foreground=[("disabled", self.MUTED)],
        )

    def _build_widgets(self) -> None:
        outer = tk.Frame(self.root, bg=self.BG, padx=18, pady=14)
        outer.pack(fill="both", expand=True)

        header = tk.Frame(outer, bg=self.BG)
        header.pack(fill="x", pady=(0, 10))
        tk.Label(
            header,
            text="Gaussian touch model · Mahalanobis decoder · language-aware hit regions",
            bg=self.BG,
            fg=self.TEAL,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Smart Touch Keyboard",
            bg=self.BG,
            fg=self.TEXT,
            font=("Segoe UI", 19, "bold"),
        ).pack(anchor="w", pady=(2, 0))
        tk.Label(
            header,
            text=(
                "Move the mouse to preview a touch; click to type. Clicks in the "
                "gaps are decoded probabilistically."
            ),
            bg=self.BG,
            fg=self.MUTED,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(3, 0))

        self._build_output_panel(outer)

        main = tk.Frame(outer, bg=self.BG)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self._build_keyboard_panel(main)
        self._build_inspector(main)

    def _build_output_panel(self, parent: tk.Widget) -> None:
        output_frame = tk.Frame(
            parent,
            bg=self.PANEL,
            highlightbackground=self.LINE,
            highlightthickness=1,
        )
        output_frame.pack(fill="x", pady=(0, 12))

        row = tk.Frame(output_frame, bg=self.PANEL)
        row.pack(fill="x", padx=12, pady=(8, 3))
        tk.Label(
            row,
            text="DECODED TEXT",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left")
        self.char_count_var = tk.StringVar(value="0 characters")
        tk.Label(
            row,
            textvariable=self.char_count_var,
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 9),
        ).pack(side="right")

        self.output_var = tk.StringVar(value="")
        tk.Label(
            output_frame,
            textvariable=self.output_var,
            bg="#07111a",
            fg=self.TEXT,
            anchor="w",
            justify="left",
            wraplength=1100,
            padx=12,
            pady=13,
            font=("Consolas", 17),
            height=2,
        ).pack(fill="x", padx=10, pady=(0, 10))

    def _build_keyboard_panel(self, parent: tk.Widget) -> None:
        keyboard_panel = tk.Frame(
            parent,
            bg=self.PANEL,
            highlightbackground=self.LINE,
            highlightthickness=1,
        )
        keyboard_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        tk.Label(
            keyboard_panel,
            text="PROBABILISTIC KEYBOARD",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", padx=12, pady=(10, 4))

        self.canvas = tk.Canvas(
            keyboard_panel,
            bg="#182936",
            highlightthickness=0,
            height=345,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self.canvas.bind("<Configure>", self.callbacks["resize"])
        self.canvas.bind("<Motion>", self.callbacks["motion"])
        self.canvas.bind("<Leave>", self.callbacks["leave"])
        self.canvas.bind("<Button-1>", self.callbacks["click"])

        legend = tk.Frame(keyboard_panel, bg=self.PANEL)
        legend.pack(fill="x", padx=12, pady=(3, 8))
        self._legend_item(legend, self.TEAL, "1σ Gaussian contour")
        self._legend_item(
            legend,
            self.ORANGE,
            "2σ bounding rectangle (overlap is normal)",
            dashed=True,
        )
        self._legend_item(legend, "#71899a", "Physical keycap")
        self._legend_item(legend, self.BLUE, "Most likely next letter")

        controls = tk.Frame(keyboard_panel, bg=self.PANEL)
        controls.pack(fill="x", padx=12, pady=(0, 7))
        ttk.Checkbutton(
            controls,
            text="Gaussian contours",
            variable=self.show_gaussians,
            command=self.callbacks["refresh"],
        ).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(
            controls,
            text="2σ bounding boxes",
            variable=self.show_boxes,
            command=self.callbacks["refresh"],
        ).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(
            controls,
            text="Language model",
            variable=self.use_language,
            command=self.callbacks["refresh"],
        ).pack(side="left", padx=(0, 10))

        button_frame = tk.Frame(keyboard_panel, bg=self.PANEL)
        button_frame.pack(fill="x", padx=12, pady=(0, 10))
        self._button(button_frame, "Calibrate", self.callbacks["calibrate"]).pack(
            side="left", padx=(0, 6)
        )
        self._button(button_frame, "Reset model", self.callbacks["reset"]).pack(
            side="left", padx=(0, 6)
        )
        self._button(button_frame, "Undo", self.callbacks["undo"]).pack(
            side="left", padx=(0, 6)
        )
        self._button(button_frame, "Clear text", self.callbacks["clear"]).pack(
            side="left"
        )

        self.calibration_var = tk.StringVar(
            value="Calibration is optional. Initial model uses key geometry as a prior."
        )
        tk.Label(
            keyboard_panel,
            textvariable=self.calibration_var,
            bg=self.PANEL_2,
            fg="#b9d9d2",
            anchor="w",
            justify="left",
            padx=10,
            pady=8,
            font=("Segoe UI", 9),
            wraplength=850,
        ).pack(fill="x", padx=10, pady=(0, 10))

    def _build_inspector(self, parent: tk.Widget) -> None:
        inspector = tk.Frame(
            parent,
            bg=self.PANEL,
            width=290,
            highlightbackground=self.LINE,
            highlightthickness=1,
        )
        inspector.grid(row=0, column=1, sticky="nsew")
        inspector.grid_propagate(False)

        tk.Label(
            inspector,
            text="LIVE MODEL INSPECTOR",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", padx=12, pady=(10, 7))
        self.inspector_var = tk.StringVar(
            value="Move over a letter key or the gaps between keys."
        )
        tk.Label(
            inspector,
            textvariable=self.inspector_var,
            bg=self.PANEL_2,
            fg=self.TEXT,
            justify="left",
            anchor="nw",
            wraplength=255,
            padx=10,
            pady=10,
            font=("Consolas", 9),
        ).pack(fill="x", padx=10)

        tk.Label(
            inspector,
            text="POSTERIOR CANDIDATES",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", padx=12, pady=(12, 5))
        candidate_frame = tk.Frame(inspector, bg=self.PANEL)
        candidate_frame.pack(fill="x", padx=12)
        self.candidate_labels: List[tk.Label] = []
        for _ in range(5):
            label = tk.Label(
                candidate_frame,
                text="—",
                bg=self.PANEL,
                fg=self.TEXT,
                anchor="w",
                font=("Consolas", 9),
            )
            label.pack(fill="x", pady=2)
            self.candidate_labels.append(label)

        tk.Label(
            inspector,
            text="LANGUAGE CONTEXT",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", padx=12, pady=(13, 5))
        self.context_var = tk.StringVar(
            value="Prefix: ∅\nNext letter: —\nPrior: —"
        )
        tk.Label(
            inspector,
            textvariable=self.context_var,
            bg=self.PANEL_2,
            fg=self.BLUE,
            justify="left",
            anchor="nw",
            wraplength=255,
            padx=10,
            pady=10,
            font=("Consolas", 9),
        ).pack(fill="x", padx=10)

        tk.Label(
            inspector,
            text="MODEL NOTES",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", padx=12, pady=(13, 5))
        notes = (
            "• Grey rectangles - Physical keycaps.\n"
            "• 1σ ellipse - probability contour.\n"
            "• Orange box - (not click boundary) encloses the 2σ ellipse.\n"
            "• The decoder scores all letters, including taps in gaps.\n"
            "• Blue outline marks the most likely next letter."
        )
        tk.Label(
            inspector,
            text=notes,
            bg=self.PANEL_2,
            fg=self.MUTED,
            justify="left",
            anchor="nw",
            wraplength=255,
            padx=10,
            pady=10,
            font=("Segoe UI", 9),
        ).pack(fill="x", padx=10)

    def _button(
        self, parent: tk.Widget, label: str, command: Callable
    ) -> ttk.Button:
        return ttk.Button(parent, text=label, command=command)

    def _legend_item(
        self,
        parent: tk.Frame,
        color: str,
        text: str,
        dashed: bool = False,
    ) -> None:
        marker = tk.Canvas(
            parent, width=22, height=12, bg=self.PANEL, highlightthickness=0
        )
        marker.pack(side="left", padx=(0, 5))
        marker.create_line(
            1,
            6,
            21,
            6,
            fill=color,
            width=2,
            dash=(3, 2) if dashed else (),
        )
        tk.Label(
            parent,
            text=text,
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 8),
        ).pack(side="left", padx=(0, 13))

    # ------------------------------------------------------------------
    # Display updates
    # ------------------------------------------------------------------
    def set_output(self, text: str) -> None:
        self.output_var.set(text)
        self.char_count_var.set(f"{len(text)} characters")

    def set_calibration_status(self, message: str) -> None:
        self.calibration_var.set(message)

    def update_inspector(
        self,
        prefix: str,
        priors: Dict[str, float],
        language_enabled: bool,
        current_result: Optional[dict],
        last_touch_label: str,
    ) -> None:
        next_letter = max(priors, key=priors.get)
        next_percent = priors[next_letter] * 100
        context = f"Prefix: {prefix or '∅ (word start)'}\n"
        context += (
            f"Next letter: {next_letter.upper()}\n"
            if prefix and language_enabled
            else "Next letter: —\n"
        )
        context += (
            f"Prior: {next_percent:.1f}%"
            if prefix and language_enabled
            else "Prior: uniform (no prefix)"
        )
        self.context_var.set(context)

        if not current_result:
            self.inspector_var.set(
                last_touch_label + "\nMove over the keyboard to preview scores."
            )
            for label in self.candidate_labels:
                label.configure(text="—")
            return

        result = current_result
        winner = result["winner"]
        key = winner["key"]
        self.inspector_var.set(
            f"TOUCH: ({result['x']:.1f}, {result['y']:.1f}) px\n"
            f"PREFIX USED: {result['prefix'] or '∅'}\n"
            f"WINNER: {winner['letter'].upper()}  ({winner['posterior'] * 100:.1f}%)\n"
            f"Mahalanobis d²: {winner['d2']:.3f}\n"
            f"μ: ({key['mu_x']:.1f}, {key['mu_y']:.1f})\n"
            f"σx / σy: {winner['sx']:.1f} / {winner['sy']:.1f} px\n"
            f"Δ: ({winner['dx']:.1f}, {winner['dy']:.1f})\n"
            "Decision: spatial likelihood × language prior"
        )

        for index, label in enumerate(self.candidate_labels):
            if index < len(result["candidates"]):
                item = result["candidates"][index]
                label.configure(
                    text=(
                        f"{index + 1}.  {item['letter'].upper()}      "
                        f"{item['posterior'] * 100:6.2f}%"
                    )
                )
            else:
                label.configure(text="—")

    # ------------------------------------------------------------------
    # Canvas drawing
    # ------------------------------------------------------------------
    def _draw_round_rect(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        radius: float,
        **kwargs,
    ) -> None:
        radius = max(2.0, min(radius, (x1 - x0) / 2, (y1 - y0) / 2))
        points = [
            x0 + radius,
            y0,
            x1 - radius,
            y0,
            x1,
            y0,
            x1,
            y0 + radius,
            x1,
            y1 - radius,
            x1,
            y1,
            x1 - radius,
            y1,
            x0 + radius,
            y1,
            x0,
            y1,
            x0,
            y1 - radius,
            x0,
            y0 + radius,
            x0,
            y0,
        ]
        self.canvas.create_polygon(
            points, smooth=True, splinesteps=12, **kwargs
        )

    def redraw(
        self,
        model: KeyboardModel,
        text: str,
        shift: bool,
        calibrating: bool,
        calibration_index: int,
        current_result: Optional[dict],
    ) -> None:
        """Render the physical keys, Gaussian contours, boxes and highlights"""
        self.canvas.delete("all")
        if not model.keys:
            return

        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self._draw_round_rect(
            6,
            6,
            width - 6,
            height - 6,
            18,
            fill="#1b2d3a",
            outline="#344b5a",
            width=1,
        )

        prefix = model.current_prefix(text)
        language_enabled = self.use_language.get()
        priors = model.next_letter_distribution(prefix, language_enabled)
        max_prior = max(priors.values())

        candidate_rank = {}
        if current_result:
            for index, candidate in enumerate(current_result["candidates"][:3]):
                candidate_rank[candidate["letter"]] = index

        context_letter = (
            max(priors, key=priors.get) if language_enabled and prefix else None
        )

        # Physical keycaps
        for letter, key in model.keys.items():
            rank = candidate_rank.get(letter)
            if rank == 0:
                fill, outline, line_width = "#285d59", "#ffcf83", 2
            elif rank in (1, 2):
                fill, outline, line_width = "#294653", "#6ec9bd", 2
            else:
                fill, outline, line_width = "#354956", "#617989", 1

            target_is_letter = (
                calibrating
                and calibration_index < len(CALIBRATION_PHRASE)
                and CALIBRATION_PHRASE[calibration_index] == letter
            )
            if target_is_letter:
                fill, outline, line_width = "#6a5030", self.ORANGE, 3

            self._draw_round_rect(
                key["x0"],
                key["y0"],
                key["x1"],
                key["y1"],
                7,
                fill=fill,
                outline=outline,
                width=line_width,
            )

        for key in model.specials:
            fill, outline = "#2a3d4a", "#617989"
            target_is_space = (
                calibrating
                and calibration_index < len(CALIBRATION_PHRASE)
                and CALIBRATION_PHRASE[calibration_index] == " "
                and key["action"] == "space"
            )
            if target_is_space:
                fill, outline = "#6a5030", self.ORANGE
            self._draw_round_rect(
                key["x0"],
                key["y0"],
                key["x1"],
                key["y1"],
                7,
                fill=fill,
                outline=outline,
                width=1,
            )

        # Gaussian contours and 2σ axis-aligned bounding rectangles.
        for letter, key in model.keys.items():
            scale = model.context_scale(
                priors[letter], max_prior, language_enabled, bool(prefix)
            )
            display_key = dict(key)
            display_key["sx"] *= scale
            display_key["sy"] *= scale

            if self.show_gaussians.get():
                points = model.ellipse_points(display_key, 1.0)
                self.canvas.create_line(
                    points,
                    fill=self.TEAL,
                    width=1.1,
                    smooth=True,
                )

            if self.show_boxes.get():
                radius_x = 2.0 * display_key["sx"]
                radius_y = 2.0 * display_key["sy"]
                self.canvas.create_rectangle(
                    display_key["mu_x"] - radius_x,
                    display_key["mu_y"] - radius_y,
                    display_key["mu_x"] + radius_x,
                    display_key["mu_y"] + radius_y,
                    outline=self.ORANGE,
                    width=1,
                    dash=(4, 4),
                )

        # Blue outline indicates most likely next letter
        if context_letter in model.keys:
            key = model.keys[context_letter]
            self.canvas.create_rectangle(
                key["x0"] - 3,
                key["y0"] - 3,
                key["x1"] + 3,
                key["y1"] + 3,
                outline=self.BLUE,
                width=2,
                dash=(2, 2),
            )

        for letter, key in model.keys.items():
            label = letter.upper() if shift else letter
            self.canvas.create_text(
                key["cx"],
                key["cy"],
                text=label,
                fill="#f2f7fa",
                font=("Segoe UI", 15, "bold"),
            )
        for key in model.specials:
            self.canvas.create_text(
                key["cx"],
                key["cy"],
                text=key["label"],
                fill="#d5e0e7",
                font=("Segoe UI", 10, "bold"),
            )

        if current_result:
            x, y = current_result["x"], current_result["y"]
            self.canvas.create_oval(
                x - 6, y - 6, x + 6, y + 6, outline="#ffffff", width=2
            )
            self.canvas.create_line(
                x - 12, y, x + 12, y, fill="#ffffff", width=1
            )
            self.canvas.create_line(
                x, y - 12, x, y + 12, fill="#ffffff", width=1
            )
