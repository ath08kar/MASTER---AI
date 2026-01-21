"""
Orchestrator Agent
------------------
Purpose:
- Combine signals from Market Researcher + Technical ML Agent
- Produce a final, explainable trading decision

Design Rules:
- NO ML models
- NO file I/O
- NO external dependencies
- Deterministic & explainable logic
"""

from typing import List, Dict


def orchestrator_decision(
    market_regime: str,
    market_confidence: float,
    tech_probs: List[float],
    min_confidence: float = 0.6
) -> Dict[str, str]:
    """
    Final decision logic.

    Parameters
    ----------
    market_regime : str
        'Bullish', 'Bearish', or 'Sideways'
    market_confidence : float
        Confidence score from Market Researcher (0–1)
    tech_probs : list
        [P(DOWN), P(HOLD), P(UP)] from Technical ML Agent
    min_confidence : float
        Minimum probability threshold to act

    Returns
    -------
    dict
        {
            'decision': 'BUY' | 'SELL' | 'HOLD' | 'WAIT',
            'reason': str
        }
    """

    # ----------------------------
    # Validate inputs (defensive)
    # ----------------------------
    if not isinstance(tech_probs, (list, tuple)) or len(tech_probs) != 3:
        return {
            "decision": "WAIT",
            "reason": "Invalid technical probability input"
        }

    try:
        p_down, p_hold, p_up = [float(p) for p in tech_probs]
    except Exception:
        return {
            "decision": "WAIT",
            "reason": "Non-numeric technical probabilities"
        }

    # ----------------------------
    # RULE 1: Strong Bearish Market
    # ----------------------------
    if market_regime == "Bearish" and market_confidence >= 0.6:
        return {
            "decision": "HOLD",
            "reason": "Market regime is bearish with high confidence"
        }

    # ----------------------------
    # RULE 2: High-Confidence BUY
    # ----------------------------
    if (
        market_regime == "Bullish"
        and market_confidence >= 0.6
        and p_up >= min_confidence
        and p_up > max(p_down, p_hold)
    ):
        return {
            "decision": "BUY",
            "reason": (
                f"Bullish market regime and strong UP probability "
                f"({p_up:.2f})"
            )
        }

    # ----------------------------
    # RULE 3: High-Confidence SELL
    # ----------------------------
    if (
        p_down >= min_confidence
        and p_down > max(p_up, p_hold)
    ):
        return {
            "decision": "SELL",
            "reason": (
                f"High DOWN probability detected "
                f"({p_down:.2f})"
            )
        }

    # ----------------------------
    # RULE 4: Default Safety
    # ----------------------------
    return {
        "decision": "WAIT",
        "reason": "Signals are mixed or confidence is insufficient"
    }
