import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ModelConfig:
    name: str = "llama-3.3-70b-versatile"
    provider: str = "groq"
    temperature: float = 0.7
    max_tokens: int = 1024
    api_key: Optional[str] = None


@dataclass
class AppConfig:
    groq_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY")
    )
    model_a: ModelConfig = field(init=False)
    model_b: ModelConfig = field(init=False)
    model_c: ModelConfig = field(init=False)
    results_dir: str = "results"
    judge_model: str = field(
        default_factory=lambda: os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile")
    )
    judge_temperature: float = 0.0
    max_retries: int = 2
    async_workers: int = 5

    def __post_init__(self):
        self.model_a = ModelConfig(
            name=os.getenv("GROQ_MODEL_A", "llama-3.3-70b-versatile"),
            temperature=0.7,
            api_key=self.groq_api_key,
        )
        self.model_b = ModelConfig(
            name=os.getenv("GROQ_MODEL_B", "llama-3.3-70b-versatile"),
            temperature=0.7,
            api_key=self.groq_api_key,
        )
        self.model_c = ModelConfig(
            name=os.getenv("GROQ_MODEL_C", "llama-3.3-70b-versatile"),
            temperature=0.7,
            api_key=self.groq_api_key,
        )


def load_config(path: Optional[str] = None) -> AppConfig:
    from dotenv import load_dotenv

    load_dotenv()

    if path and os.path.exists(path):
        with open(path) as f:
            if path.endswith(".json"):
                overrides = json.load(f)
            else:
                import yaml
                overrides = yaml.safe_load(f)
        cfg = AppConfig()
        for k, v in overrides.get("config", {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        for model_key in ("model_a", "model_b", "model_c"):
            m_overrides = overrides.get(model_key, {})
            existing = getattr(cfg, model_key)
            for mk, mv in m_overrides.items():
                if hasattr(existing, mk):
                    setattr(existing, mk, mv)
        return cfg

    return AppConfig()
