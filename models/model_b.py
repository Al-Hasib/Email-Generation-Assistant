"""
Model B: Role-Playing Only via LangChain

Uses a ChatPromptTemplate with a strong persona prompt directly
chained to the LLM. No few-shot examples, no graph — just role-playing
to test whether persona alone can match the structured approach.
"""

from langchain_groq import ChatGroq

from .base import BaseModel, EmailInput
from src.prompt_templates import role_playing_prompt


class ModelB(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.llm = ChatGroq(
            model=config.name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
        )
        self.chain = role_playing_prompt | self.llm

    def name(self) -> str:
        return self.config.name

    def strategy_name(self) -> str:
        return "Role-Playing Only (LangChain)"

    def generate(self, email_input: EmailInput) -> tuple:
        facts_bullets = self._facts_bullets(email_input)
        result = self.chain.invoke({
            "intent": email_input.intent,
            "facts_bullets": facts_bullets,
            "tone": email_input.tone,
        })
        raw = result.content
        return raw, raw.strip()
