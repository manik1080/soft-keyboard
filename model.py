"""Probability, geometry, language-prior and calibration logic.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

LETTERS = "abcdefghijklmnopqrstuvwxyz"
WORD_LIMIT = 50000
CHAR_NGRAM_ORDER = 3
PREFIX_BACKOFF_MIX = 0.025

CALIBRATION_PHRASE = (
    "the quick brown fox jumps over the lazy dog "
    "pack my box with five dozen liquor jugs"
)
LANGUAGE_WEIGHT = 0.55


def normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, weights.get(ch, 0.0)) for ch in LETTERS)
    if total <= 0.0:
        return {ch: 1.0 / len(LETTERS) for ch in LETTERS}
    return {ch: max(0.0, weights.get(ch, 0.0)) / total for ch in LETTERS}


def sample_stats(samples: List[Tuple[float, float]]) -> Optional[dict]:
    """Return unbiased sample mean/covariance for (dx, dy) samples."""
    if len(samples) < 2:
        return None
    n = len(samples)
    mean_x = sum(point[0] for point in samples) / n
    mean_y = sum(point[1] for point in samples) / n
    var_x = sum((point[0] - mean_x) ** 2 for point in samples) / (n - 1)
    var_y = sum((point[1] - mean_y) ** 2 for point in samples) / (n - 1)
    cov_xy = sum(
        (point[0] - mean_x) * (point[1] - mean_y) for point in samples
    ) / (n - 1)
    return {
        "mx": mean_x,
        "my": mean_y,
        "vx": var_x,
        "vy": var_y,
        "cov": cov_xy,
    }


class FrequencyPrefixLanguageModel:
    """Frequency-weighted next-character model built from wordfreq's English list.

    For every word w with corpus probability f(w), each observed prefix p gets
    f(w) added to the bucket for the character immediately following p.
    This yields P(next_character | typed_prefix) after normalization.

    Character 1-, 2- and 3-gram tables provide backoff for prefixes absent from
    the word list. They are weighted by the same corpus word probabilities.
    """

    def __init__(self, max_words: int = WORD_LIMIT) -> None:
        try:
            from wordfreq import get_frequency_dict, top_n_list
        except ImportError as exc:
            raise RuntimeError(
                "The frequency language model requires wordfreq. Install the "
                "project dependencies with: python -m pip install -r requirements.txt"
            ) from exc

        ranked_words = top_n_list(
            "en", max_words, wordlist="best", ascii_only=True
        )
        frequencies = get_frequency_dict("en", wordlist="best")

        self.prefix_next: Dict[str, Dict[str, float]] = {}
        self.ngram_next: Dict[str, Dict[str, float]] = {}
        self.character_weights: Dict[str, float] = {ch: 0.0 for ch in LETTERS}
        self.words_loaded = 0

        for raw_word in ranked_words:
            word = raw_word.lower()
            if not word or not word.isascii() or not word.isalpha():
                continue

            frequency = float(frequencies.get(word, 0.0))
            if frequency <= 0.0:
                continue

            self.words_loaded += 1
            for index, next_char in enumerate(word):
                prefix = word[:index]
                bucket = self.prefix_next.setdefault(prefix, {})
                bucket[next_char] = bucket.get(next_char, 0.0) + frequency
                self.character_weights[next_char] += frequency

                for order in range(1, min(CHAR_NGRAM_ORDER, index) + 1):
                    context = word[index - order:index]
                    ngram_bucket = self.ngram_next.setdefault(context, {})
                    ngram_bucket[next_char] = (
                        ngram_bucket.get(next_char, 0.0) + frequency
                    )

        if self.words_loaded < 20:
            raise RuntimeError(
                "wordfreq returned too few usable English words"
            )

    @staticmethod
    def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(0.0, weights.get(ch, 0.0)) for ch in LETTERS)
        if total <= 0.0:
            return {ch: 1.0 / len(LETTERS) for ch in LETTERS}
        return {
            ch: max(0.0, weights.get(ch, 0.0)) / total
            for ch in LETTERS
        }

    def _backoff_distribution(self, prefix: str) -> Dict[str, float]:
        for order in range(min(CHAR_NGRAM_ORDER, len(prefix)), 0, -1):
            context = prefix[-order:]
            counts = self.ngram_next.get(context)
            if counts:
                return self._normalize(counts)
        return self._normalize(self.character_weights)

    def next_letter_distribution(self, prefix: str) -> Dict[str, float]:
        prefix = prefix.lower()
        if not prefix:
            return {ch: 1.0 / len(LETTERS) for ch in LETTERS}

        direct_counts = self.prefix_next.get(prefix)
        backoff = self._backoff_distribution(prefix)
        if not direct_counts:
            return backoff

        direct_total = sum(direct_counts.values())
        # Add a small backoff component so sparse/rare prefixes remain smooth.
        # Scale it against direct evidence instead of using a fixed arbitrary count.
        weights = {
            ch: direct_counts.get(ch, 0.0)
            + PREFIX_BACKOFF_MIX * direct_total * backoff[ch]
            for ch in LETTERS
        }
        return self._normalize(weights)


_FREQUENCY_LANGUAGE_MODEL: Optional[FrequencyPrefixLanguageModel] = None


def get_frequency_language_model() -> FrequencyPrefixLanguageModel:
    """Load the language resource once and share it across model instances."""
    global _FREQUENCY_LANGUAGE_MODEL
    if _FREQUENCY_LANGUAGE_MODEL is None:
        _FREQUENCY_LANGUAGE_MODEL = FrequencyPrefixLanguageModel()
    return _FREQUENCY_LANGUAGE_MODEL


class KeyboardModel:
    """Owns keyboard geometry and all probabilistic/model state."""

    def __init__(self) -> None:
        self.keys: Dict[str, dict] = {}
        self.specials: List[dict] = []
        self.calibrated = False
        self.calibration_samples: Dict[str, List[Tuple[float, float]]] = {
            ch: [] for ch in LETTERS
        }
        self.calibration_global: List[Tuple[float, float]] = []
        self.language_model = get_frequency_language_model()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @staticmethod
    def _make_key(
        label: str,
        x: float,
        y: float,
        width: float,
        height: float,
        kind: str = "letter",
        action: Optional[str] = None,
    ) -> dict:
        key = {
            "label": label,
            "x0": x,
            "y0": y,
            "x1": x + width,
            "y1": y + height,
            "cx": x + width / 2,
            "cy": y + height / 2,
            "w": width,
            "h": height,
            "kind": kind,
            "action": action,
        }
        if kind == "letter":
            key.update(
                {
                    "letter": label,
                    "mu_x": key["cx"],
                    "mu_y": key["cy"],
                    "base_sx": max(9.0, width * 0.29),
                    "base_sy": max(10.0, height * 0.34),
                    "sx": max(9.0, width * 0.29),
                    "sy": max(10.0, height * 0.34),
                    "rho": 0.0,
                }
            )
        return key

    def build_geometry(self, width: int, height: int) -> None:
        """Build QWERTY key rectangles for the current canvas dimensions."""
        width = max(600, int(width))
        height = max(280, int(height))
        base_width, base_gap, base_height = 68.0, 9.0, 57.0
        max_row_width = 10 * base_width + 9 * base_gap
        scale = min(1.0, (width - 28) / max_row_width)
        key_width = base_width * scale
        gap = base_gap * scale
        key_height = base_height * scale
        row_gap = 13.0 * scale
        top = max(15.0, (height - (4 * key_height + 3 * row_gap)) / 2)

        self.keys = {}
        self.specials = []

        def place_row(items: List[Tuple[str, float, str]], y: float) -> None:
            total = sum(multiplier * key_width for _, multiplier, _ in items)
            total += (len(items) - 1) * gap
            x = (width - total) / 2
            for label, multiplier, action in items:
                item_width = multiplier * key_width
                if action == "letter":
                    self.keys[label] = self._make_key(
                        label, x, y, item_width, key_height, "letter"
                    )
                else:
                    self.specials.append(
                        self._make_key(
                            label, x, y, item_width, key_height, "special", action
                        )
                    )
                x += item_width + gap

        rows = [
            [(ch, 1.0, "letter") for ch in "qwertyuiop"],
            [(ch, 1.0, "letter") for ch in "asdfghjkl"],
            [("⇧", 1.18, "shift")]
            + [(ch, 1.0, "letter") for ch in "zxcvbnm"]
            + [("⌫", 1.18, "backspace")],
            [
                ("?123", 1.12, "symbols"),
                (",", 0.78, "comma"),
                ("English (US)", 4.9, "space"),
                (".", 0.78, "period"),
                ("↵", 1.12, "enter"),
            ],
        ]
        for row_index, items in enumerate(rows):
            y = top + row_index * (key_height + row_gap)
            place_row(items, y)

        # Preserve learned offsets relative to each key's new physical centre.
        self._apply_calibration_model()

    def special_at(self, x: float, y: float) -> Optional[str]:
        for key in self.specials:
            if key["x0"] <= x <= key["x1"] and key["y0"] <= y <= key["y1"]:
                return key["action"]
        return None

    def physical_key_at(self, x: float, y: float) -> Optional[str]:
        """Return the physical keycap under a point, for diagnostics only."""
        for letter, key in self.keys.items():
            if key["x0"] <= x <= key["x1"] and key["y0"] <= y <= key["y1"]:
                return letter
        return None

    # ------------------------------------------------------------------
    # Language model and probabilistic decoding
    # ------------------------------------------------------------------
    @staticmethod
    def current_prefix(text: str) -> str:
        match = re.search(r"[a-zA-Z]+$", text)
        return match.group(0).lower() if match else ""

    def next_letter_distribution(
        self, prefix: str, use_language: bool = True
    ) -> Dict[str, float]:
        """Return corpus-frequency-weighted P(next letter | typed prefix)."""
        if not use_language or not prefix:
            return {ch: 1.0 / len(LETTERS) for ch in LETTERS}
        return self.language_model.next_letter_distribution(prefix)

    @staticmethod
    def context_scale(
        prior: float, max_prior: float, enabled: bool, has_prefix: bool
    ) -> float:
        if not enabled or not has_prefix:
            return 1.0
        ratio = max(0.0, min(1.0, prior / max(max_prior, 1e-12)))
        return 0.92 + 0.24 * math.sqrt(ratio)

    def score_touch(
        self,
        x: float,
        y: float,
        prefix: str,
        use_language: bool = True,
    ) -> dict:
        """Score all 26 letters using Gaussian likelihood × language prior."""
        if len(self.keys) != 26:
            raise RuntimeError("Keyboard geometry has not been initialized.")

        priors = self.next_letter_distribution(prefix, use_language)
        max_prior = max(priors.values())
        candidates = []

        for letter, key in self.keys.items():
            prior = priors[letter]
            scale = self.context_scale(prior, max_prior, use_language, bool(prefix))
            sigma_x = max(3.0, key["sx"] * scale)
            sigma_y = max(3.0, key["sy"] * scale)
            rho = max(-0.75, min(0.75, key["rho"]))

            cov_xx = sigma_x * sigma_x
            cov_yy = sigma_y * sigma_y
            cov_xy = rho * sigma_x * sigma_y
            determinant = max(1e-9, cov_xx * cov_yy - cov_xy * cov_xy)
            delta_x = x - key["mu_x"]
            delta_y = y - key["mu_y"]

            mahalanobis_d2 = (
                cov_yy * delta_x * delta_x
                - 2 * cov_xy * delta_x * delta_y
                + cov_xx * delta_y * delta_y
            ) / determinant
            log_likelihood = (
                -0.5 * mahalanobis_d2
                - 0.5 * math.log(determinant)
                - math.log(2 * math.pi)
            )
            log_prior = math.log(max(prior, 1e-12))
            log_score = log_likelihood + (
                LANGUAGE_WEIGHT * log_prior if use_language else 0.0
            )
            candidates.append(
                {
                    "letter": letter,
                    "key": key,
                    "prior": prior,
                    "scale": scale,
                    "d2": mahalanobis_d2,
                    "dx": delta_x,
                    "dy": delta_y,
                    "sx": sigma_x,
                    "sy": sigma_y,
                    "rho": rho,
                    "log_likelihood": log_likelihood,
                    "log_score": log_score,
                }
            )

        candidates.sort(key=lambda item: item["log_score"], reverse=True)
        max_score = candidates[0]["log_score"]
        exponentials = [
            math.exp(max(-700.0, item["log_score"] - max_score))
            for item in candidates
        ]
        total = sum(exponentials) or 1.0
        for item, value in zip(candidates, exponentials):
            item["posterior"] = value / total

        return {
            "x": x,
            "y": y,
            "prefix": prefix,
            "priors": priors,
            "candidates": candidates,
            "winner": candidates[0],
        }

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def reset_calibration(self) -> None:
        self.calibrated = False
        self.calibration_samples = {ch: [] for ch in LETTERS}
        self.calibration_global = []
        self._apply_calibration_model()

    def record_calibration_touch(self, expected_letter: str, x: float, y: float) -> None:
        key = self.keys[expected_letter]
        offset = (x - key["cx"], y - key["cy"])
        self.calibration_samples[expected_letter].append(offset)
        self.calibration_global.append(offset)

    def fit_calibration(self) -> dict:
        """Fit per-key means/covariances and return a small training summary."""
        self.calibrated = len(self.calibration_global) >= 10
        self._apply_calibration_model()
        counts = {
            ch: len(self.calibration_samples[ch])
            for ch in LETTERS
            if self.calibration_samples[ch]
        }
        return {
            "sample_count": len(self.calibration_global),
            "per_key_counts": counts,
            "calibrated": self.calibrated,
        }

    def _apply_calibration_model(self) -> None:
        if not self.keys:
            return
        global_stats = sample_stats(self.calibration_global)
        if not self.calibrated or not global_stats:
            for key in self.keys.values():
                key["mu_x"], key["mu_y"] = key["cx"], key["cy"]
                key["sx"], key["sy"], key["rho"] = (
                    key["base_sx"], key["base_sy"], 0.0
                )
            return

        global_var_x = max(9.0, global_stats["vx"])
        global_var_y = max(9.0, global_stats["vy"])
        global_cov = global_stats["cov"]

        for letter, key in self.keys.items():
            samples = self.calibration_samples[letter]
            local = sample_stats(samples)
            count = len(samples)
            blend = count / (count + 4.0)

            local_mean_x = local["mx"] if local else global_stats["mx"]
            local_mean_y = local["my"] if local else global_stats["my"]
            key["mu_x"] = (
                key["cx"]
                + blend * local_mean_x
                + (1 - blend) * global_stats["mx"]
            )
            key["mu_y"] = (
                key["cy"]
                + blend * local_mean_y
                + (1 - blend) * global_stats["my"]
            )

            local_var_x = local["vx"] if local else global_var_x
            local_var_y = local["vy"] if local else global_var_y
            local_cov = local["cov"] if local else global_cov
            mixed_var_x = blend * local_var_x + (1 - blend) * global_var_x
            mixed_var_y = blend * local_var_y + (1 - blend) * global_var_y
            mixed_cov = blend * local_cov + (1 - blend) * global_cov

            fitted_var_x = 0.35 * key["base_sx"] ** 2 + 0.65 * mixed_var_x
            fitted_var_y = 0.35 * key["base_sy"] ** 2 + 0.65 * mixed_var_y
            fitted_var_x = min(
                key["base_sx"] ** 2 * 2.0,
                max(key["base_sx"] ** 2 * 0.45, fitted_var_x),
            )
            fitted_var_y = min(
                key["base_sy"] ** 2 * 2.0,
                max(key["base_sy"] ** 2 * 0.45, fitted_var_y),
            )
            key["sx"] = math.sqrt(fitted_var_x)
            key["sy"] = math.sqrt(fitted_var_y)
            denominator = math.sqrt(max(1.0, fitted_var_x * fitted_var_y))
            empirical_rho = mixed_cov / denominator
            key["rho"] = max(-0.35, min(0.35, 0.25 * empirical_rho))

    # ------------------------------------------------------------------
    # Visualization math (kept here because it derives from covariance)
    # ------------------------------------------------------------------
    @staticmethod
    def ellipse_points(
        key: dict, n_sigma: float = 1.0, steps: int = 48
    ) -> List[float]:
        """Return flattened x/y coordinates for a covariance ellipse."""
        sigma_x, sigma_y, rho = key["sx"], key["sy"], key["rho"]
        a = sigma_x * sigma_x
        b = rho * sigma_x * sigma_y
        d = sigma_y * sigma_y
        trace = a + d
        root = math.sqrt(max(0.0, (a - d) ** 2 + 4 * b * b))
        lambda_1 = max(1e-6, (trace + root) / 2.0)
        lambda_2 = max(1e-6, (trace - root) / 2.0)
        angle = 0.5 * math.atan2(2 * b, a - d)
        cos_angle, sin_angle = math.cos(angle), math.sin(angle)
        points: List[float] = []
        for index in range(steps + 1):
            theta = 2 * math.pi * index / steps
            u = n_sigma * math.sqrt(lambda_1) * math.cos(theta)
            v = n_sigma * math.sqrt(lambda_2) * math.sin(theta)
            px = key["mu_x"] + u * cos_angle - v * sin_angle
            py = key["mu_y"] + u * sin_angle + v * cos_angle
            points.extend((px, py))
        return points
