import os
import time

from tqdm.auto import tqdm
from rag_helper import RAGBase


# ---------------------------------------------------------------------------
# Cost
#
# NOTE: these are the course's placeholder prices (for gpt-5.4-mini). z.ai /
# glm-5.2 pricing differs, so treat the dollar figures as illustrative. The
# token counts themselves are exact. Swap these constants for real z.ai prices
# when you want accurate billing numbers.
# ---------------------------------------------------------------------------
INPUT_PRICE_PER_MILLION = 0.75
OUTPUT_PRICE_PER_MILLION = 4.50


def calc_price(usage):
    """Cost of a single call, from its token usage object."""
    input_cost = (usage.input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (usage.output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_price(usages):
    total_cost = 0.0

    for usage in usages:
        cost = calc_price(usage)
        total_cost = total_cost + cost["total_cost"]

    return total_cost


# ---------------------------------------------------------------------------
# Structured output via Anthropic tool-use
# ---------------------------------------------------------------------------
def _pydantic_to_tool(output_type):
    """Turn a pydantic model into an Anthropic tool definition.

    The tool's `input_schema` is exactly the JSON schema of the model, so when
    the model is forced to call the tool it must produce fields that validate
    against our pydantic class.
    """
    schema = output_type.model_json_schema()
    return {
        "name": output_type.__name__,
        "description": "Return the result as a structured object matching this schema.",
        "input_schema": schema,
    }


def _extract_tool_input(response, tool_name):
    """Find the tool_use block for `tool_name` and return its input dict."""
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
            return block.input
    raise RuntimeError(f"Model did not call the '{tool_name}' tool.")


def llm_structured(client, instructions, user_prompt, output_type, model=None):
    """Call the model and parse the response into `output_type` (a pydantic class).

    Returns (parsed_object, usage).
    """
    model = model or os.getenv("ANTHROPIC_MODEL", "glm-5.2")
    tool = _pydantic_to_tool(output_type)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=instructions,
        messages=[{"role": "user", "content": user_prompt}],
        tools=[tool],
        # Force the model to call our schema-tool, guaranteeing structured output.
        tool_choice={"type": "tool", "name": tool["name"]},
    )

    raw_input = _extract_tool_input(response, tool["name"])
    parsed = output_type.model_validate(raw_input)

    return parsed, response.usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model=None,
    max_retries=3,
):
    """llm_structured with exponential backoff on failure (rate limits, etc.)."""
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# RAG that records token usage per call
# ---------------------------------------------------------------------------
class RAGWithUsage(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    # Slightly different boosts than RAGBase: these are the values we ended up
    # with after the search-tuning step (answer field weighted highest).
    def search(self, query, num_results=5):
        boost_dict = {"question": 1.0, "answer": 2.0, "section": 0.1}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def llm(self, prompt):
        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.instructions,
            messages=[{"role": "user", "content": prompt}],
        )

        self.last_usage = response.usage
        self.usages.append(response.usage)

        return response.content[0].text

    def total_cost(self):
        return calc_total_price(self.usages)


# ---------------------------------------------------------------------------
# Parallel map with a progress bar
# ---------------------------------------------------------------------------
def map_progress(pool, seq, f):
    """Submit `f(el)` for every `el` in `seq` to a ThreadPoolExecutor `pool`,
    showing a tqdm bar. Returns results in the original order."""
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results
