"""
Custom Evaluation Metrics for Email Generation Assistant.

Metric 1: Fact Recall Accuracy (FRA)
  - Measures what proportion of required key facts are present in the
    generated email.
  - Implementation: Each fact is split into meaningful keywords (stopwords
    removed). A fact is considered "recalled" if >= 50% of its keywords
    appear in the email.
  - Score: 0.0 to 1.0 (fraction of facts successfully recalled).

Metric 2: Tone Adherence Score (TAS)
  - Measures how well the email matches the requested tone.
  - Implementation: LLM-as-a-Judge — an evaluator model rates how well
    the email's language, formality, and style match the target tone
    on a scale of 1-5, normalized to 0.0-1.0.
  - Score: 0.0 to 1.0.

Metric 3: Overall Quality Score (OQS)
  - Measures holistic email quality including structure, clarity,
    grammar, professionalism, and call-to-action effectiveness.
  - Implementation: LLM-as-a-Judge — an evaluator model provides a
    composite quality rating on a scale of 1-5, normalized to 0.0-1.0.
  - Score: 0.0 to 1.0.
"""

import re
from typing import List, Tuple

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from src.prompt_templates import TONE_EVALUATION_PROMPT, QUALITY_EVALUATION_PROMPT
from evaluation.test_scenarios import TestScenario


class _JudgeScore(BaseModel):
    score: int = Field(ge=1, le=5, description="Score from 1 (lowest) to 5 (highest)")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "about", "up", "it",
    "its", "this", "that", "these", "those", "i", "me", "my", "we",
    "our", "you", "your", "he", "him", "his", "she", "her", "they",
    "them", "their", "what", "which", "who", "whom", "please", "let",
}


def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())


def extract_keywords(text: str) -> set:
    tokens = re.findall(r'[a-z0-9]+', normalize(text))
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


# =============================================================================
# Metric 1: Fact Recall Accuracy
# =============================================================================

FACT_KEYWORD_THRESHOLD = 0.5


def _fact_in_email(fact: str, email: str) -> bool:
    fact_kw = extract_keywords(fact)
    email_kw = extract_keywords(email)
    if not fact_kw:
        return True
    overlap = fact_kw & email_kw
    ratio = len(overlap) / len(fact_kw)
    return ratio >= FACT_KEYWORD_THRESHOLD


def metric_fact_recall_accuracy(email: str, scenario: TestScenario) -> float:
    facts = scenario.facts
    if not facts:
        return 1.0
    recalled = sum(1 for f in facts if _fact_in_email(f, email))
    return recalled / len(facts)


def metric_fact_recall_accuracy_detail(
    email: str, scenario: TestScenario
) -> Tuple[float, List[dict]]:
    details = []
    for f in scenario.facts:
        fact_kw = extract_keywords(f)
        email_kw = extract_keywords(email)
        overlap = fact_kw & email_kw
        ratio = len(overlap) / len(fact_kw) if fact_kw else 1.0
        found = ratio >= FACT_KEYWORD_THRESHOLD
        details.append({
            "fact": f,
            "found": found,
            "keyword_match_ratio": round(ratio, 2),
            "missing_keywords": sorted(fact_kw - email_kw),
        })
    score = sum(d["found"] for d in details) / len(details) if details else 1.0
    return score, details


# =============================================================================
# Metric 2: Tone Adherence Score (LLM-as-a-Judge)
# =============================================================================


def _llm_judge_score(prompt_template, llm: BaseChatModel, **kwargs) -> float:
    structured_llm = llm.with_structured_output(_JudgeScore)
    result = (prompt_template | structured_llm).invoke(kwargs)
    return (result.score - 1) / 4


def metric_tone_adherence(
    email: str, scenario: TestScenario, llm: BaseChatModel
) -> float:
    return _llm_judge_score(
        TONE_EVALUATION_PROMPT, llm,
        tone=scenario.tone, email=email,
    )


# =============================================================================
# Metric 3: Overall Quality Score (LLM-as-a-Judge)
# =============================================================================


def metric_overall_quality(
    email: str, llm: BaseChatModel
) -> float:
    return _llm_judge_score(
        QUALITY_EVALUATION_PROMPT, llm,
        email=email,
    )
