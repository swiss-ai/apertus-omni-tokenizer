"""A minimal stand-in for the parts of vLLM the reasoning parser imports.

Used by tests/test_parsers.py only when vLLM is not installed (CI, a laptop), so
the parser's logic is still exercised. ``BaseThinkingReasoningParser`` mirrors
upstream ``vllm/reasoning/basic_parsers.py`` (token ids from the vocab, and the
streaming split the parser falls back to); keep it in step with upstream.
"""

import sys
import types
from dataclasses import dataclass


@dataclass
class DeltaMessage:
    reasoning: str | None = None
    content: str | None = None


class ReasoningParserManager:
    modules: dict = {}

    @classmethod
    def register_module(cls, name):
        def wrap(parser_cls):
            cls.modules[name] = parser_cls
            return parser_cls

        return wrap


class BaseThinkingReasoningParser:
    def __init__(self, tokenizer, *args, **kwargs):
        self.model_tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.start_token_id = self.vocab[self.start_token]
        self.end_token_id = self.vocab[self.end_token]

    def extract_reasoning_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ):
        if len(delta_token_ids) == 1 and delta_token_ids[0] in (self.start_token_id, self.end_token_id):
            return None
        if self.start_token_id in previous_token_ids:
            if self.end_token_id in delta_token_ids:
                end_index = delta_text.find(self.end_token)
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(reasoning=delta_text[:end_index], content=content or None)
            if self.end_token_id in previous_token_ids:
                return DeltaMessage(content=delta_text)
            return DeltaMessage(reasoning=delta_text)
        if self.start_token_id in delta_token_ids:
            if self.end_token_id in delta_token_ids:
                start_index = delta_text.find(self.start_token)
                end_index = delta_text.find(self.end_token)
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=delta_text[start_index + len(self.start_token) : end_index],
                    content=content or None,
                )
            return DeltaMessage(reasoning=delta_text)
        return DeltaMessage(content=delta_text)


def install() -> None:
    """Register the stub modules under their vLLM import paths."""
    reasoning = types.ModuleType("vllm.reasoning")
    reasoning.ReasoningParserManager = ReasoningParserManager
    basic = types.ModuleType("vllm.reasoning.basic_parsers")
    basic.BaseThinkingReasoningParser = BaseThinkingReasoningParser
    protocol = types.ModuleType("vllm.entrypoints.generate.base.protocol")
    protocol.DeltaMessage = DeltaMessage
    for name in ("vllm", "vllm.entrypoints", "vllm.entrypoints.generate", "vllm.entrypoints.generate.base"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vllm.reasoning"] = reasoning
    sys.modules["vllm.reasoning.basic_parsers"] = basic
    sys.modules["vllm.entrypoints.generate.base.protocol"] = protocol
