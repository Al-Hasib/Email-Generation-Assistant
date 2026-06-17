"""
Model C: Self-Reflection Agent (LangGraph)

Same Plan → Write → Review → (accept/rewrite) structure as Model A,
but uses stricter review prompts and enforces up to 2 rewrites.
Designed to maximise quality through iterative self-improvement.
"""

from typing import Literal

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

from .base import BaseModel, EmailInput, GraphState
from src.prompt_templates import (
    planner_prompt, writer_prompt, rewriter_prompt,
    SELF_REFLECT_REVIEWER_PROMPT,
)


def _build_graph(llm: ChatGroq) -> StateGraph:

    def plan_node(state: GraphState) -> dict:
        chain = planner_prompt | llm
        r = chain.invoke({
            "intent": state["intent"],
            "facts_bullets": state["facts_bullets"],
            "tone": state["tone"],
        })
        return {"plan": r.content}

    def write_node(state: GraphState) -> dict:
        chain = writer_prompt | llm
        r = chain.invoke({
            "plan": state["plan"],
            "intent": state["intent"],
            "facts_bullets": state["facts_bullets"],
            "tone": state["tone"],
        })
        return {"email": r.content}

    def review_node(state: GraphState) -> dict:
        chain = SELF_REFLECT_REVIEWER_PROMPT | llm
        r = chain.invoke({
            "email": state["email"],
            "facts_bullets": state["facts_bullets"],
            "tone": state["tone"],
        })
        return {"feedback": r.content}

    def rewrite_node(state: GraphState) -> dict:
        chain = rewriter_prompt | llm
        r = chain.invoke({
            "feedback": state["feedback"],
            "email": state["email"],
            "plan": state["plan"],
            "intent": state["intent"],
            "facts_bullets": state["facts_bullets"],
            "tone": state["tone"],
        })
        return {"email": r.content, "rewrite_count": state.get("rewrite_count", 0) + 1}

    def review_router(state: GraphState) -> Literal["rewrite", "end"]:
        fb = state.get("feedback", "").strip().upper()
        if fb.startswith("ACCEPT"):
            return "end"
        if state.get("rewrite_count", 0) < 2:
            return "rewrite"
        return "end"

    builder = StateGraph(GraphState)
    builder.add_node("plan", plan_node)
    builder.add_node("write", write_node)
    builder.add_node("review", review_node)
    builder.add_node("rewrite", rewrite_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "write")
    builder.add_edge("write", "review")
    builder.add_conditional_edges("review", review_router, {
        "rewrite": "rewrite",
        "end": END,
    })
    builder.add_edge("rewrite", "review")

    return builder.compile()


class ModelC(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        llm = ChatGroq(
            model=config.name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
        )
        self.graph = _build_graph(llm)

    def name(self) -> str:
        return self.config.name

    def strategy_name(self) -> str:
        return "Self-Reflection Agent (LangGraph)"

    def generate(self, email_input: EmailInput) -> tuple:
        result = self.graph.invoke({
            "intent": email_input.intent,
            "facts_bullets": self._facts_bullets(email_input),
            "tone": email_input.tone,
            "plan": "",
            "email": "",
            "rewrite_count": 0,
            "feedback": "",
        })
        raw = result.get("email", "")
        return raw, raw.strip()
