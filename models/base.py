from abc import ABC, abstractmethod
from dataclasses import dataclass
import operator
from typing import Annotated, List, TypedDict


@dataclass
class EmailInput:
    intent: str
    facts: List[str]
    tone: str


@dataclass
class EmailResult:
    input: EmailInput
    raw_output: str
    parsed_email: str
    model_name: str
    strategy_name: str


class GraphState(TypedDict):
    intent: str
    facts_bullets: str
    tone: str
    plan: str
    email: str
    rewrite_count: Annotated[int, operator.add]
    feedback: str


class BaseModel(ABC):
    def __init__(self, config: "ModelConfig"):  # noqa: F821
        self.config = config

    @abstractmethod
    def generate(self, email_input: EmailInput) -> tuple:
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def strategy_name(self) -> str:
        ...

    def _facts_bullets(self, email_input: EmailInput) -> str:
        return "\n".join(f"- {f}" for f in email_input.facts)
