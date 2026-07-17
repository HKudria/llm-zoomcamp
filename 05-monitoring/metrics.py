"""
Instrument the RAG pipeline so every LLM call is captured as a record.

Adapted from the course's `metrics.py` to the Anthropic SDK. Two things
change versus the OpenAI version:

  * OpenAI's `usage.total_tokens` does not exist on the Anthropic usage
    object. Anthropic gives `input_tokens` and `output_tokens`, so we
    compute `total_tokens = input + output`.
  * The call itself goes through `client.messages.create(system=..., ...)`
    instead of `client.responses.create(...)`, and the answer text lives in
    `response.content[0].text` instead of `response.output_text`.

Everything else — the `LLMCallRecord` dataclass, the timing, the
`RAGWithMetrics` subclass that only overrides `llm()` — is identical to the
course version. We inherit `rag()` for free, so every answer gets logged.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime

from rag_helper import RAGBase


@dataclass
class LLMCallRecord:
    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int        # API name; = Anthropic input_tokens
    completion_tokens: int    # API name; = Anthropic output_tokens
    total_tokens: int         # input + output (Anthropic has no .total_tokens)
    response_time: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Cost
#
# Price per 1,000,000 tokens, keyed by a substring of the model name. The
# course uses gpt-5.4-mini's placeholder rates. We keep those as the default
# so the numbers match the lesson, but real z.ai / glm pricing differs —
# treat the dollars as illustrative; the token counts are exact.
# ---------------------------------------------------------------------------
PRICE_TABLE = {
    # model-substring : (input $/M, output $/M)
    "gpt-5.4-mini": (0.15, 0.60),
    "glm-5.2":      (0.15, 0.60),   # placeholder; swap for real z.ai prices
}


def calculate_cost(model, usage):
    """Cost of one call from its usage object. Returns 0 for unknown models."""
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    for key, (in_price, out_price) in PRICE_TABLE.items():
        if key in model:
            return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
    return 0


class RAGWithMetrics(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_call: LLMCallRecord = None

    def llm(self, prompt):
        start_time = time.time()
        response = self._call_llm(prompt)
        response_time = time.time() - start_time
        self._log_response(prompt, response, response_time)
        # Anthropic: the assistant reply is a list of content blocks; the
        # first text block holds a plain-text answer.
        return response.content[0].text

    def _call_llm(self, prompt):
        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.instructions,
            messages=[{"role": "user", "content": prompt}],
        )
        return response

    def _log_response(self, prompt, response, response_time):
        usage = response.usage
        cost = calculate_cost(self.model, usage)

        call_record = LLMCallRecord(
            model=self.model,
            prompt=prompt,
            instructions=self.instructions,
            answer=response.content[0].text,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            response_time=response_time,
            cost=cost,
        )

        print(call_record)
        self.last_call = call_record
