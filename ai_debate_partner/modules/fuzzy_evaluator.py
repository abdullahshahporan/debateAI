from __future__ import annotations

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from .utils import clamp_score


class DebateFuzzyEvaluator:
    def __init__(self) -> None:
        universe = np.arange(0, 10.1, 0.1)

        self.logical_strength = ctrl.Antecedent(universe, "logical_strength")
        self.evidence_usage = ctrl.Antecedent(universe, "evidence_usage")
        self.clarity = ctrl.Antecedent(universe, "clarity")
        self.emotional_bias = ctrl.Antecedent(universe, "emotional_bias")
        self.fallacy_level = ctrl.Antecedent(universe, "fallacy_level")
        self.time_efficiency = ctrl.Antecedent(universe, "time_efficiency")
        self.relevance = ctrl.Antecedent(universe, "relevance")
        self.debate_score = ctrl.Consequent(universe, "debate_score")

        self._define_membership_functions()
        self._define_rules()

    def _define_membership_functions(self) -> None:
        self.logical_strength["low"] = fuzz.trapmf(self.logical_strength.universe, [0, 0, 2, 4])
        self.logical_strength["medium"] = fuzz.trimf(self.logical_strength.universe, [3, 5, 7])
        self.logical_strength["high"] = fuzz.trapmf(self.logical_strength.universe, [6, 8, 10, 10])

        self.evidence_usage["weak"] = fuzz.trapmf(self.evidence_usage.universe, [0, 0, 2, 4])
        self.evidence_usage["moderate"] = fuzz.trimf(self.evidence_usage.universe, [3, 5, 7])
        self.evidence_usage["strong"] = fuzz.trapmf(self.evidence_usage.universe, [6, 8, 10, 10])

        self.clarity["unclear"] = fuzz.trapmf(self.clarity.universe, [0, 0, 2, 4])
        self.clarity["average"] = fuzz.trimf(self.clarity.universe, [3, 5, 7])
        self.clarity["clear"] = fuzz.trapmf(self.clarity.universe, [6, 8, 10, 10])

        self.emotional_bias["low"] = fuzz.trapmf(self.emotional_bias.universe, [0, 0, 2, 4])
        self.emotional_bias["medium"] = fuzz.trimf(self.emotional_bias.universe, [3, 5, 7])
        self.emotional_bias["high"] = fuzz.trapmf(self.emotional_bias.universe, [6, 8, 10, 10])

        self.fallacy_level["low"] = fuzz.trapmf(self.fallacy_level.universe, [0, 0, 2, 4])
        self.fallacy_level["medium"] = fuzz.trimf(self.fallacy_level.universe, [3, 5, 7])
        self.fallacy_level["high"] = fuzz.trapmf(self.fallacy_level.universe, [6, 8, 10, 10])

        self.time_efficiency["poor"] = fuzz.trapmf(self.time_efficiency.universe, [0, 0, 2, 4])
        self.time_efficiency["good"] = fuzz.trimf(self.time_efficiency.universe, [3, 5, 7])
        self.time_efficiency["excellent"] = fuzz.trapmf(self.time_efficiency.universe, [6, 8, 10, 10])

        self.relevance["low"] = fuzz.trapmf(self.relevance.universe, [0, 0, 2, 4])
        self.relevance["medium"] = fuzz.trimf(self.relevance.universe, [3, 5, 7])
        self.relevance["high"] = fuzz.trapmf(self.relevance.universe, [6, 8, 10, 10])

        self.debate_score["poor"] = fuzz.trapmf(self.debate_score.universe, [0, 0, 2, 4])
        self.debate_score["average"] = fuzz.trimf(self.debate_score.universe, [3, 5, 7])
        self.debate_score["excellent"] = fuzz.trapmf(self.debate_score.universe, [6, 8, 10, 10])

    def _define_rules(self) -> None:
        rules = [
            ctrl.Rule(
                self.logical_strength["high"] & self.evidence_usage["strong"] & self.clarity["clear"] & self.relevance["high"],
                self.debate_score["excellent"],
            ),
            ctrl.Rule(
                self.logical_strength["medium"] & self.evidence_usage["moderate"] & self.clarity["average"],
                self.debate_score["average"],
            ),
            ctrl.Rule(
                self.logical_strength["low"] | self.evidence_usage["weak"] | self.clarity["unclear"],
                self.debate_score["poor"],
            ),
            ctrl.Rule(
                self.emotional_bias["high"] | self.fallacy_level["high"],
                self.debate_score["poor"],
            ),
            ctrl.Rule(
                self.time_efficiency["excellent"] & self.relevance["high"] & self.logical_strength["high"],
                self.debate_score["excellent"],
            ),
            ctrl.Rule(
                self.fallacy_level["low"] & self.emotional_bias["low"] & self.evidence_usage["strong"],
                self.debate_score["excellent"],
            ),
            ctrl.Rule(
                self.relevance["low"],
                self.debate_score["poor"],
            ),
            ctrl.Rule(
                self.time_efficiency["poor"] & self.clarity["unclear"],
                self.debate_score["poor"],
            ),
            ctrl.Rule(
                self.logical_strength["medium"] & self.clarity["clear"] & self.evidence_usage["moderate"],
                self.debate_score["average"],
            ),
            ctrl.Rule(
                self.logical_strength["high"] & self.evidence_usage["weak"],
                self.debate_score["average"],
            ),
        ]
        self.system = ctrl.ControlSystem(rules)
        self.simulator = ctrl.ControlSystemSimulation(self.system)

    def evaluate(self, feature_scores: dict[str, float]) -> float:
        mapping = {
            "logical_strength": feature_scores.get("Logical Strength", 0.0),
            "evidence_usage": feature_scores.get("Evidence Usage", 0.0),
            "clarity": feature_scores.get("Clarity", 0.0),
            "emotional_bias": feature_scores.get("Emotional Bias", 0.0),
            "fallacy_level": feature_scores.get("Fallacy Level", 0.0),
            "time_efficiency": feature_scores.get("Time Efficiency", 0.0),
            "relevance": feature_scores.get("Relevance", 0.0),
        }

        for key, value in mapping.items():
            self.simulator.input[key] = clamp_score(value)

        self.simulator.compute()
        return clamp_score(self.simulator.output["debate_score"])


_EVALUATOR = DebateFuzzyEvaluator()


def calculate_debate_score(feature_scores: dict[str, float]) -> float:
    """Compute the fuzzy debate score from the extracted features."""
    try:
        return round(_EVALUATOR.evaluate(feature_scores), 1)
    except Exception as exc:
        raise RuntimeError(f"Fuzzy evaluation failed: {exc}") from exc
