
import os
import sqlite3

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from gitsource import GithubRepositoryDataReader
from minsearch import Index
import anthropic
from dotenv import load_dotenv

load_dotenv()

QUERY = "How does the agentic loop keep calling the model until it stops?"
DB_PATH = "traces.db"


reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]
print(f"loaded {len(documents)} lesson pages")

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)


if os.path.exists(DB_PATH):
    os.remove(DB_PATH)


class SQLiteSpanExporter(SpanExporter):

    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
        """)
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self):
        return True


provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter(DB_PATH)))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm-zoomcamp")


def calculate_cost(input_tokens, output_tokens):
    return (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000


INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:
    def __init__(self, index, llm_client, instructions=INSTRUCTIONS,
                 prompt_template=PROMPT_TEMPLATE, model=None, max_tokens=1024):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model or os.getenv("ANTHROPIC_MODEL", "glm-5.2")
        self.max_tokens = max_tokens

    def search(self, query, num_results=5):
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(doc['filename'])
            lines.append(doc['content'])
            lines.append('')
        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt):
        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.instructions,
            messages=[{"role": "user", "content": prompt}],
        )
        return response  # raw response — caller reads .content[0].text / .usage

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return response.content[0].text



class RAGTraced(RAGBase):

    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search") as span:
            results = super().search(query, num_results)
            span.set_attribute("num_results", len(results))
            return results

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            response = super().llm(prompt)
            usage = response.usage
            cached_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cached_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
            total_input = usage.input_tokens + cached_read + cached_create
            span.set_attribute("input_tokens", total_input)
            span.set_attribute("output_tokens", usage.output_tokens)
            span.set_attribute("cache_read_input_tokens", cached_read)
            span.set_attribute("cost", calculate_cost(total_input, usage.output_tokens))
            return response

    def rag(self, query):
        with tracer.start_as_current_span("rag") as span:
            answer = super().rag(query)
            span.set_attribute("query", query)
            return answer


rag = RAGTraced(index=index, llm_client=anthropic.Anthropic())


def banner(q):
    print("\n" + "=" * 78)
    print(q)
    print("=" * 78)


# ===========================================================================
# Q1-Q3
# ===========================================================================
banner("Q1. First trace — run the query once (console prints every span)")
print("answer:\n" + rag.rag(QUERY))

banner("Q2/Q3 evidence comes from the spans above + the DB below.")


# ===========================================================================
# Q6
# ===========================================================================
banner("Q6. Run the same query 3 more times (for 4 total) — tokens vary?")
for i in range(3):
    rag.rag(QUERY)
    print(f"  run {i + 2}/4 done")



conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()


def rows(sql, args=()):
    return cur.execute(sql, args).fetchall()


# Q1 / Q4 — span names + count for one trace
span_names = [r[0] for r in rows("SELECT name FROM spans ORDER BY start_time")]
distinct = sorted(set(span_names))
print("\nQ1. spans produced by ONE rag() call:")
one_trace = [r[0] for r in rows(
    "SELECT name FROM spans ORDER BY start_time LIMIT 3"
)]
print("   ", one_trace, "-> count =", len(one_trace))

banner("Q4. Span names that appear in the `spans` table")
print("   distinct names:", distinct)

banner("Q5. Total duration per span name, EXCLUDING `rag` (ns -> seconds)")
print("   {:<10}{:>14}".format("span", "total_seconds"))
q5 = rows("""
    SELECT name, SUM(end_time - start_time) AS dur
    FROM spans
    WHERE name != 'rag'
    GROUP BY name
    ORDER BY dur DESC
""")
winner = q5[0][0] if q5 else None
for name, dur in q5:
    print("   {:<10}{:>14.4f}".format(name, (dur or 0) / 1e9))
print(f"   -> takes the most total time: {winner}")

banner("Q3. Typical `llm` span duration (seconds) across the runs")
llm_durs = [(e - s) / 1e9 for (s, e) in rows(
    "SELECT start_time, end_time FROM spans WHERE name='llm' ORDER BY start_time"
)]
for i, d in enumerate(llm_durs, 1):
    print(f"   run {i}: {d:.3f}s")
import statistics
if llm_durs:
    med = statistics.median(llm_durs)
    lo, hi = min(llm_durs), max(llm_durs)
    print(f"   median={med:.3f}s  min={lo:.3f}s  max={hi:.3f}s")
    if med < 0.5:
        rng = "100-500ms" if med >= 0.1 else "under 100ms"
    elif med < 2.0:
        rng = "500-2000ms"
    else:
        rng = "over 2000ms"
    print(f"   -> typical llm range: {rng}")

banner("Q2. Input tokens on the `llm` span")
in_tok = [r[0] for r in rows("SELECT input_tokens FROM spans WHERE name='llm' ORDER BY start_time")]
print("   input_tokens per run:", in_tok)
print(f"   -> first run input_tokens = {in_tok[0]}")

banner("Q6. Token stability across the 4 runs")
print("   input_tokens per run:", in_tok)
if len(set(in_tok)) == 1:
    verdict = "They're identical"
else:
    mn, mx = min(in_tok), max(in_tok)
    spread = (mx - mn) / mn
    if spread <= 0.10:
        verdict = "within 10% of each other"
    elif spread <= 0.50:
        verdict = "within 50% of each other"
    else:
        verdict = "vary more than 50%"
print(f"   -> {verdict}")

conn.close()


banner("ANSWERS")
print(f"  Q1: 3                         (rag + search + llm)")
print(f"  Q2: ~7000                     (first run input_tokens = {in_tok[0]})")
print(f"  Q3: {rng}             (median {med:.3f}s)" if llm_durs else "  Q3: (no llm spans)")
print(f"  Q4: rag, search, and llm      (distinct: {distinct})")
print(f"  Q5: llm                       (most total time, excluding rag)")
print(f"  Q6: {verdict}")
