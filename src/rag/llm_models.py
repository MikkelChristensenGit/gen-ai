from __future__ import annotations

from enum import StrEnum, auto

from langchain_openai import ChatOpenAI

from rag.settings import settings


class LLMModel(StrEnum):
    REPHRASER = auto()
    EXPANDER = auto()
    CHAT = auto()


# Alias so call sites can use llm_model.REPHRASER
llm_model = LLMModel

MODEL_CATALOG: dict[LLMModel, str] = {
    LLMModel.REPHRASER: settings.REPHRASER_MODEL,
    LLMModel.EXPANDER: settings.EXPANDER_MODEL,
    LLMModel.CHAT: settings.CHAT_MODEL,
}


def resolve_model_name(model_role: LLMModel) -> str:
    try:
        return MODEL_CATALOG[model_role]
    except KeyError as exc:
        raise ValueError(f"Unsupported LLM model role: {model_role}") from exc


def build_chat_llm(model_role: LLMModel, *, temperature: float = 0.0) -> ChatOpenAI:
    model_name = resolve_model_name(model_role)
    return ChatOpenAI(model=model_name, temperature=temperature)
