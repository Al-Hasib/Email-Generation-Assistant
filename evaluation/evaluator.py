"""
Evaluation Runner: Runs all test scenarios through a model and
computes all 3 custom metrics. Supports async parallel execution.
"""

import sys
import os
import asyncio
from dataclasses import dataclass
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from models.base import EmailInput, EmailResult
from evaluation.test_scenarios import SCENARIOS, TestScenario
from evaluation.metrics import (
    metric_fact_recall_accuracy_detail,
    metric_tone_adherence,
    metric_overall_quality,
)
from src.email_generator import generate_email


@dataclass
class MetricDetail:
    fact_recall_score: float
    fact_recall_details: List[dict]
    tone_adherence_score: float
    overall_quality_score: float


@dataclass
class ScenarioResult:
    scenario_id: int
    intent: str
    tone: str
    email_result: EmailResult
    metrics: MetricDetail


def _evaluate_single(model, scenario: TestScenario, judge_llm: ChatGroq) -> ScenarioResult:
    email_input = EmailInput(
        intent=scenario.intent,
        facts=scenario.facts,
        tone=scenario.tone,
    )
    email_result = generate_email(model, email_input)
    parsed = email_result.parsed_email

    fra_score, fra_details = metric_fact_recall_accuracy_detail(parsed, scenario)
    tas_score = metric_tone_adherence(parsed, scenario, judge_llm)
    oqs_score = metric_overall_quality(parsed, judge_llm)

    return ScenarioResult(
        scenario_id=scenario.id,
        intent=scenario.intent,
        tone=scenario.tone,
        email_result=email_result,
        metrics=MetricDetail(
            fact_recall_score=fra_score,
            fact_recall_details=fra_details,
            tone_adherence_score=tas_score,
            overall_quality_score=oqs_score,
        ),
    )


def run_evaluation(model, config) -> List[ScenarioResult]:
    judge_llm = ChatGroq(
        model=config.judge_model,
        temperature=config.judge_temperature,
        api_key=config.groq_api_key,
    )
    results: List[ScenarioResult] = []
    n = len(SCENARIOS)
    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"    [{i}/{n}] {scenario.intent[:50]}...")
        results.append(_evaluate_single(model, scenario, judge_llm))
    return results


async def run_evaluation_async(model, config) -> List[ScenarioResult]:
    judge_llm = ChatGroq(
        model=config.judge_model,
        temperature=config.judge_temperature,
        api_key=config.groq_api_key,
    )

    async def run_one(scenario: TestScenario) -> ScenarioResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _evaluate_single, model, scenario, judge_llm
        )

    sem = asyncio.Semaphore(config.async_workers)

    async def bounded(scenario: TestScenario) -> ScenarioResult:
        async with sem:
            return await run_one(scenario)

    tasks = [bounded(s) for s in SCENARIOS]
    return await asyncio.gather(*tasks)
