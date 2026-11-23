"""Application controller and entry point for Smart Touch Keyboard.

Run from this directory with: python app.py
"""
from __future__ import annotations

import tkinter as tk
from typing import Dict, List, Optional, Tuple

from model import CALIBRATION_PHRASE, KeyboardModel
from ui import KeyboardUI


class SmartTouchKeyboardApp:
    """Coordinates user input, model operations and UI refreshes."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.model = KeyboardModel()

        # Application/session state belongs here, not in the model or UI.
        self.text = ""
        self.undo_stack: List[str] = []
        self.shift = False
        self.calibrating = False
        self.calibration_index = 0
        self.current_result: Optional[dict] = None
        self.last_touch_label = "No touch yet"
        self._resize_after_id = None

        callbacks = {
            "resize": self.on_canvas_resize,
            "motion": self.on_motion,
            "leave": self.on_leave,
            "click": self.on_click,
            "refresh": self.refresh,
            "calibrate": self.toggle_calibration,
            "reset": self.reset_model,
            "undo": self.undo,
            "clear": self.clear_text,
        }
        self.ui = KeyboardUI(root, callbacks)

        self.root.after(150, self._initialize_geometry)

    # ------------------------------------------------------------------
    # Layout and refresh
    # ------------------------------------------------------------------
    def _initialize_geometry(self) -> None:
        self.model.build_geometry(
            self.ui.canvas.winfo_width(), self.ui.canvas.winfo_height()
        )
        self.refresh()

    def on_canvas_resize(self, event) -> None:
        """Debounce resize events so geometry is rebuilt only after resizing settles."""
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(
            30, lambda width=event.width, height=event.height: self._apply_geometry(
                width, height
            )
        )

    def _apply_geometry(self, width: int, height: int) -> None:
        self._resize_after_id = None
        self.model.build_geometry(width, height)
        self.refresh()

    def refresh(self) -> None:
        """Update all visible UI elements from the latest app/model state."""
        self.ui.set_output(self.text)
        prefix = self.model.current_prefix(self.text)
        language_enabled = self.ui.use_language.get()
        priors = self.model.next_letter_distribution(prefix, language_enabled)

        self.ui.update_inspector(
            prefix=prefix,
            priors=priors,
            language_enabled=language_enabled,
            current_result=self.current_result,
            last_touch_label=self.last_touch_label,
        )
        self.ui.redraw(
            model=self.model,
            text=self.text,
            shift=self.shift,
            calibrating=self.calibrating,
            calibration_index=self.calibration_index,
            current_result=self.current_result,
        )

    # ------------------------------------------------------------------
    # Pointer interaction
    # ------------------------------------------------------------------
    def on_motion(self, event) -> None:
        if self.calibrating:
            return

        x, y = float(event.x), float(event.y)
        special = self.model.special_at(x, y)
        if special:
            self.current_result = None
            self.last_touch_label = f"Pointer over special key: {special}"
        else:
            prefix = self.model.current_prefix(self.text)
            self.current_result = self.model.score_touch(
                x,
                y,
                prefix,
                use_language=self.ui.use_language.get(),
            )
            self.last_touch_label = "Touch preview"
        self.refresh()

    def on_leave(self, _event=None) -> None:
        if not self.calibrating:
            self.current_result = None
            self.refresh()

    def on_click(self, event) -> None:
        x, y = float(event.x), float(event.y)
        special = self.model.special_at(x, y)

        if self.calibrating:
            self._handle_calibration_click(x, y, special)
            return

        if special:
            self._handle_special_action(special)
            return

        prefix = self.model.current_prefix(self.text)
        result = self.model.score_touch(
            x,
            y,
            prefix,
            use_language=self.ui.use_language.get(),
        )
        winner = result["winner"]["letter"]

        self.undo_stack.append(self.text)
        self.text += winner.upper() if self.shift else winner
        self.shift = False

        # Clear the old candidate preview: it was scored using the prefix before
        # this character was committed. The next pointer movement scores afresh.
        self.current_result = None
        self.last_touch_label = f"Last touch decoded as {winner.upper()}"
        self.refresh()

    def _handle_special_action(self, action: str) -> None:
        if action == "shift":
            self.shift = not self.shift
        elif action == "backspace":
            if self.text:
                self.undo_stack.append(self.text)
                self.text = self.text[:-1]
        elif action == "space":
            self.undo_stack.append(self.text)
            self.text += " "
        elif action == "enter":
            self.undo_stack.append(self.text)
            self.text += "\n"
        elif action == "comma":
            self.undo_stack.append(self.text)
            self.text += ","
        elif action == "period":
            self.undo_stack.append(self.text)
            self.text += "."
        elif action == "symbols":
            self.undo_stack.append(self.text)
            self.text += "?"

        self.current_result = None
        self.last_touch_label = f"Special key: {action}"
        self.refresh()

    # ------------------------------------------------------------------
    # Calibration workflow
    # ------------------------------------------------------------------
    def toggle_calibration(self) -> None:
        if self.calibrating:
            self.calibrating = False
            self.ui.set_calibration_status(
                "Calibration stopped before completion; the model was not refitted."
            )
            self.refresh()
            return

        self.model.reset_calibration()
        self.calibrating = True
        self.calibration_index = 0
        self.text = ""
        self.undo_stack.clear()
        self.current_result = None
        self.last_touch_label = "Calibration in progress"
        self._update_calibration_status()
        self.refresh()

    def _update_calibration_status(self) -> None:
        if not self.calibrating:
            return
        target = CALIBRATION_PHRASE[self.calibration_index]
        shown = "SPACE" if target == " " else target.upper()
        self.ui.set_calibration_status(
            f"CALIBRATING  {self.calibration_index + 1}/{len(CALIBRATION_PHRASE)}  ·  "
            f"Aim for {shown}; letter-area touches and near-misses are training samples."
        )

    def _handle_calibration_click(
        self, x: float, y: float, special: Optional[str]
    ) -> None:
        expected = CALIBRATION_PHRASE[self.calibration_index]

        if expected == " ":
            if special != "space":
                self.ui.set_calibration_status(
                    "Target is SPACE — click the wide spacebar."
                )
                return
            self.text += " "
        else:
            if special is not None:
                self.ui.set_calibration_status(
                    f"Target is {expected.upper()} — click in the letter area, "
                    "not on a special key."
                )
                return

            # The expected prompted character is the supervised label. Store the
            # actual coordinate even if the click is in a gap or on a neighbour.
            self.model.record_calibration_touch(expected, x, y)
            self.text += expected

        self.calibration_index += 1
        if self.calibration_index >= len(CALIBRATION_PHRASE):
            self._finish_calibration()
        else:
            self._update_calibration_status()
        self.current_result = None
        self.refresh()

    def _finish_calibration(self) -> None:
        self.calibrating = False
        summary = self.model.fit_calibration()
        counts = ", ".join(
            f"{letter.upper()}:{count}"
            for letter, count in summary["per_key_counts"].items()
        )
        status = (
            f"Calibration complete: {summary['sample_count']} letter samples. "
            f"Per-key sample counts: {counts}. "
        )
        status += (
            "Personalized model fitted."
            if summary["calibrated"]
            else "Not enough samples; using geometric prior."
        )
        self.ui.set_calibration_status(status)
        self.last_touch_label = "Calibration complete"

    # ------------------------------------------------------------------
    # Text and model controls
    # ------------------------------------------------------------------
    def undo(self) -> None:
        if self.undo_stack:
            self.text = self.undo_stack.pop()
            self.current_result = None
            self.last_touch_label = "Undo"
            self.refresh()

    def clear_text(self) -> None:
        self.undo_stack.append(self.text)
        self.text = ""
        self.current_result = None
        self.last_touch_label = "Text cleared"
        self.refresh()

    def reset_model(self) -> None:
        self.calibrating = False
        self.calibration_index = 0
        self.model.reset_calibration()
        self.current_result = None
        self.last_touch_label = "Model reset"
        self.ui.set_calibration_status(
            "Model reset. Means are at physical key centres; covariance uses "
            "conservative geometric estimates."
        )
        self.refresh()


def main() -> None:
    root = tk.Tk()
    SmartTouchKeyboardApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
