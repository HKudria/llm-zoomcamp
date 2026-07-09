"""Builds the 4 evaluation notebooks (Anthropic/z.ai adapted).

Run:  uv run python build_notebooks.py
Then execute each notebook with jupyter nbconvert --execute.

Keeps notebook JSON clean by defining cells as simple (type, source) tuples.
"""
import nbformat as nbf


def nb(cells):
    """cells: list of (kind, source) where kind in {'md','code'}."""
    out = nbf.v4.new_notebook()
    for kind, src in cells:
        if kind == "md":
            out.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            out.cells.append(nbf.v4.new_code_cell(src))
    return out


# ===========================================================================
# Notebook 1 — Generating Ground Truth Data
# ===========================================================================
data_gen = nb([
    ("md", """# 1. Generating Ground Truth Data

**Goal of this stage:** turn *subjective* quality ("is search good?") into *measurable* quality.

To measure retrieval we need labelled pairs **(question -> correct document)**.
We have no real user logs yet, so we **generate them synthetically with the LLM**:

> For each FAQ document, ask the LLM to write 5 questions that this document
> would answer. The document we generated them from is, by construction, the
> correct answer for those questions.

So we get hundreds of test questions for free, each with a known-correct doc id.
This is the foundation for **every** metric in the module."""),
    ("code", """# Load the full FAQ (several courses), then keep only llm-zoomcamp docs.
from ingest import load_faq_data
documents = load_faq_data()

documents_llm = [d for d in documents if d["course"] == "llm-zoomcamp"]
documents = documents_llm
len(documents)"""),
    ("code", """# Each document already has a stable `id` — that id becomes the LABEL.
doc = documents[0]
print(doc["id"]); print(doc["question"]); print(doc["answer"])"""),
    ("md", """## Structured output

We want a *Python object* back (a list of questions), not free text we then
parse. OpenAI offers `responses.parse(...)`; the **Anthropic** SDK achieves the
same by **forcing a tool call**: we register one tool whose `input_schema` is our
pydantic schema and force the model to call it. The tool arguments come back as a
validated dict. (See `evaluation_utils.llm_structured`.)

Define the output shape with pydantic:"""),
    ("code", """from pydantic import BaseModel

class Questions(BaseModel):
    questions: list[str]

data_gen_instructions = \"\"\"
You emulate a student who's taking our course.
Formulate 5 questions this student might ask based on a FAQ record. The record
should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the record.

The output should resemble how people ask questions
on the internet. Not too formal, not too short, not too long.
\"\"\".strip()"""),
    ("code", """# Connect to the LLM. anthropic.Anthropic() reads ANTHROPIC_AUTH_TOKEN +
# ANTHROPIC_BASE_URL (z.ai) from .env automatically.
import os, json
from dotenv import load_dotenv
import anthropic
load_dotenv()
client = anthropic.Anthropic()
print("model:", os.getenv("ANTHROPIC_MODEL"), "| base:", os.getenv("ANTHROPIC_BASE_URL"))"""),
    ("code", """# Call the structured helper for ONE document -> get a Questions object + usage.
from evaluation_utils import llm_structured, calc_price

user_prompt = json.dumps(doc)
result, usage = llm_structured(client, data_gen_instructions, user_prompt, Questions)

print(result.questions)
calc_price(usage)"""),
    ("code", """# Turn the questions into labelled ground-truth records.
import pandas as pd
records = [{"question": q, "document": doc["id"]} for q in result.questions]
pd.DataFrame(records)"""),
    ("md", """## Generate for all documents (batch + parallel)

`generate_ground_truth` wraps one doc -> 5 labelled records. We run it across
documents in parallel with `map_progress` (a ThreadPoolExecutor + tqdm bar) and
track cost.

> **Why regenerate instead of the shipped `ground_truth-new.csv`?** The live FAQ
> has drifted since that file was made (docs removed, new ones added) — ~20% of
> its doc ids no longer exist, which would cap our hit rate. **Evaluation data
> goes stale.** Regenerating against the *current* FAQ keeps the whole pipeline
> self-consistent, so we generate for all live docs here."""),
    ("code", """from concurrent.futures import ThreadPoolExecutor
from evaluation_utils import llm_structured_retry, map_progress, calc_total_price

N_GEN = len(documents)  # generate for ALL live docs so ids match the current FAQ

def generate_ground_truth(doc):
    out, usage = llm_structured_retry(
        client, data_gen_instructions, json.dumps(doc), Questions
    )
    records = [{"question": q, "document": doc["id"]} for q in out.questions]
    return records, usage

with ThreadPoolExecutor(max_workers=6) as pool:
    results = map_progress(pool, documents[:N_GEN], generate_ground_truth)

ground_truth, usages = [], []
for recs, u in results:
    ground_truth.extend(recs)
    usages.append(u)

print("generated questions:", len(ground_truth))
print("cost: $%.6f" % calc_total_price(usages))"""),
    ("code", """# Save the freshly generated ground truth. The next notebooks read THIS file.
pd.DataFrame(ground_truth).to_csv("data/ground_truth.csv", index=False)
print("saved", len(ground_truth), "rows to data/ground_truth.csv")"""),
    ("md", """**Takeaway:** we now have labelled `(question, document)` pairs. The next
notebook asks the search engine each question and checks whether it returns the
labelled document."""),
])


# ===========================================================================
# Notebook 2 — Search Evaluation (retrieval metrics)
# ===========================================================================
search_eval = nb([
    ("md", """# 2. Search Evaluation (Retrieval Metrics)

This stage answers: **does search retrieve the right document?**

It needs **no LLM** — it's pure, free, fast. We run it over the full 395-question
ground truth.

The recipe:
1. Take a ground-truth question (we know its correct doc id).
2. Run search, get the top-5 results.
3. Build a *relevance list*: `1` where a result is the correct doc, else `0`.
   e.g. correct doc at rank 1 -> `[1, 0, 0, 0, 0]`.
4. Aggregate all lists into two metrics: **Hit Rate** and **MRR**.

> We read the **freshly generated** `data/ground_truth.csv` (from notebook 1), not
> the shipped `ground_truth-new.csv` — the latter is stale vs the live FAQ (see
> notebook 1), which would deflate the metrics."""),
    ("code", """import pandas as pd
df_ground_truth = pd.read_csv("data/ground_truth.csv")
ground_truth = df_ground_truth.to_dict(orient="records")
df_ground_truth.head()"""),
    ("code", """# Build the minsearch index over the llm-zoomcamp docs.
from ingest import load_faq_data, build_index
documents = [d for d in load_faq_data() if d["course"] == "llm-zoomcamp"]
index = build_index(documents)"""),
    ("md", """A search function: `boost_dict` multiplies the score of a field. Lexical match
on the short **question** is a stronger relevance signal than a match buried in a
long **answer**, so we boost `question`."""),
    ("code", """def text_search(query):
    return index.search(query, num_results=5, boost_dict={"question": 3.0, "section": 0.5})"""),
    ("code", """# Relevance list for one question.
q = ground_truth[0]
doc_id = q["document"]
results = text_search(q["question"])
[int(d["id"] == doc_id) for d in results]  # [1,0,0,0,0] means: correct doc is at rank 1"""),
    ("code", """# Build relevance lists for ALL questions (this is the expensive-but-free part).
from tqdm.auto import tqdm

def compute_relevance(q, search_function):
    doc_id = q["document"]
    results = search_function(query=q["question"])
    return [int(d["id"] == doc_id) for d in results]

def compute_relevance_total(ground_truth, search_function):
    return [compute_relevance(q, search_function) for q in tqdm(ground_truth)]

relevance_total = compute_relevance_total(ground_truth, text_search)
relevance_total[:5]"""),
    ("md", """## Hit Rate (a.k.a. Recall@k)

Fraction of queries where the correct document appears **anywhere** in the top-k.
Did search find it at all?"""),
    ("code", """def hit_rate(relevance):
    cnt = sum(1 for line in relevance if 1 in line)
    return cnt / len(relevance)

hit_rate(relevance_total)"""),
    ("md", """## Mean Reciprocal Rank (MRR)

Like hit rate, but also rewards **position**. A hit at rank 1 scores 1.0, rank 2
scores 0.5, rank 3 scores 0.333, not-found scores 0. Averaged over queries.

> Hit Rate is the *ceiling* for MRR. MRR <= Hit Rate always. A high Hit Rate with
> a low MRR means the doc is retrieved but buried."""),
    ("code", """def mrr(relevance):
    total = 0.0
    for line in relevance:
        for rank in range(len(line)):
            if line[rank] == 1:
                total += 1 / (rank + 1)  # +1 because python is 0-indexed
                break
    return total / len(relevance)

mrr(relevance_total)"""),
    ("code", """# One reusable evaluator: takes a search function, returns both metrics.
def evaluate(ground_truth, search_function):
    rel = compute_relevance_total(ground_truth, search_function)
    return {"hit_rate": hit_rate(rel), "mrr": mrr(rel)}

evaluate(ground_truth, text_search)"""),
    ("md", """## Use the metrics to tune search

Now that quality is a number, we can **sweep parameters** and pick the best. Try a
different boost:"""),
    ("code", """def text_search_v2(query):
    return index.search(query, num_results=5, boost_dict={"question": 2.0, "section": 0.5})

evaluate(ground_truth, text_search_v2)"""),
    ("code", """# Sweep the question boost alone.
def search_boost(query, question_boost):
    return index.search(query, num_results=5, boost_dict={"question": question_boost, "section": 0.5})

for boost in [0.5, 1.0, 3.0, 5.0, 10.0]:
    res = evaluate(ground_truth, lambda query, b=boost: search_boost(query, b))
    print(f"boost={boost:>4}: {res}")"""),
    ("code", """# Full 3-parameter grid: question x answer x section boosts.
def search_boosts(query, question_boost, answer_boost, section_boost):
    return index.search(query, num_results=5, boost_dict={
        "question": question_boost, "section": section_boost, "answer": answer_boost})

results = []
for qb in [1.0, 2.0, 5.0]:
    for ab in [1.0, 2.0, 4.0, 10.0]:
        for sb in [0.1, 0.2, 0.5]:
            res = evaluate(ground_truth, lambda query, qb=qb, ab=ab, sb=sb:
                           search_boosts(query, qb, ab, sb))
            results.append({"question": qb, "answer": ab, "section": sb, **res})

df_results = pd.DataFrame(results).sort_values("mrr", ascending=False)
df_results.head(10)"""),
    ("md", """**Interpretation:** the best configs reach ~0.97 hit rate / ~0.88 MRR. Synthetic
questions tend to inflate these numbers (they're close to the FAQ wording), so
treat >95% with caution. The point of the exercise is the *method* (sweep ->
measure -> pick), which generalizes to vector search and hybrid setups."""),
])


# ===========================================================================
# Notebook 3 — Generating RAG Answers
# ===========================================================================
rag_evals = nb([
    ("md", """# 3. Generating RAG Answers (A -> Q -> A')

Retrieval metrics only test **search**. Next we test the **full RAG pipeline**:
search -> prompt -> LLM answer.

The **A -> Q -> A'** recipe:
- **A** = original FAQ answer (ground truth)
- **Q** = the synthetic question (from notebook 1)
- **A'** = the answer our RAG pipeline produces for Q

We save `(Q, A', A)` triples. The *next* notebook judges whether A' matches A."""),
    ("code", """import pandas as pd
df_ground_truth = pd.read_csv("data/ground_truth.csv")
ground_truth = df_ground_truth.to_dict(orient="records")

from ingest import load_faq_data, build_index
documents = [d for d in load_faq_data() if d["course"] == "llm-zoomcamp"]
index = build_index(documents)
doc_idx = {d["id"]: d for d in documents}  # quick id -> doc lookup"""),
    ("code", """from dotenv import load_dotenv
import anthropic
load_dotenv()
client = anthropic.Anthropic()"""),
    ("code", """# RAGWithUsage = RAGBase that records token usage per call (so we can price runs).
from evaluation_utils import RAGWithUsage
assistant = RAGWithUsage(index=index, llm_client=client, course="llm-zoomcamp")"""),
    ("code", """# Answer ONE question end-to-end through the RAG pipeline.
q = ground_truth[10]
answer = assistant.rag(q["question"])
print("cost so far: $%.6f" % assistant.total_cost())
print(answer)"""),
    ("code", "# The ground-truth answer A (from the doc the question came from):"
             "\nanswer_orig = doc_idx[q['document']]['answer']\nanswer_orig"),
    ("code", """# Build the (Q, A', A) triple.
rag_result = {
    "question": q["question"],
    "answer_llm": answer,
    "answer_orig": answer_orig,
    "document": q["document"],
}
rag_result"""),
    ("md", """## Run over many questions in parallel

> **Cost control:** set `N_RAG`. The course ran all 395 (~$0.34 with gpt-5.4-mini).
> Here we generate a live batch with glm-5.2; bump to `len(ground_truth)` for the
> full run."""),
    ("code", """from concurrent.futures import ThreadPoolExecutor
from evaluation_utils import map_progress

N_RAG = 50  # bump to len(ground_truth) for the full run
assistant.reset_usage()

def generate_rag_answer(rec):
    original_doc = doc_idx[rec["document"]]
    return {
        "question": rec["question"],
        "answer_llm": assistant.rag(rec["question"]),
        "answer_orig": original_doc["answer"],
        "document": rec["document"],
    }

with ThreadPoolExecutor(max_workers=6) as pool:
    results = map_progress(pool, ground_truth[:N_RAG], generate_rag_answer)

print("cost: $%.6f" % assistant.total_cost())"""),
    ("code", """df_results = pd.DataFrame(results)
df_results.head()
pd.DataFrame(results).to_csv("data/rag-answers.csv", index=False)
print("saved", len(results), "rows to data/rag-answers.csv")"""),
    ("md", """We now have `(Q, A', A)` triples. Time to judge whether A' is correct."""),
])


# ===========================================================================
# Notebook 4 — LLM as a Judge
# ===========================================================================
llm_judge = nb([
    ("md", """# 4. LLM as a Judge (Answer Quality)

For answer quality, exact string match is far too strict — the RAG answer need
not copy the FAQ word-for-word, it must convey the same key info.

So we use **another LLM call as the judge**: given Q, A (ground truth) and A'
(RAG answer), output `good` / `bad` **with reasoning**.

This evaluates the *entire* RAG flow in one verdict: search + prompt + LLM. The
`reasoning` field tells us *where* it broke when it's `bad`."""),
    ("code", """from pydantic import BaseModel, Field
from typing import Literal

class AnswerEvaluation(BaseModel):
    reasoning: str = Field(description="Reasoning about the quality of the answer.")
    score: Literal["good", "bad"] = Field(description="'good' if correct and complete, 'bad' otherwise.")"""),
    ("code", """aqa_judge_instructions = \"\"\"
You are an expert evaluator. You will be given:
1. A question from a student
2. The original answer from the FAQ (ground truth)
3. An answer generated by an AI assistant

Your task is to decide if the AI answer is semantically equivalent to
the original answer.

Rules:
- The AI answer does NOT need to be word-for-word identical
- It should convey the same key information
- Extra detail is fine as long as the core answer is correct
- Mark 'bad' only if the AI answer is wrong or misses the key point

Be fair and focus on correctness, not style.
\"\"\".strip()

aqa_judge_prompt = \"\"\"
Question:
{question}

Original Answer (ground truth):
{answer_orig}

AI Answer:
{answer_llm}
\"\"\".strip()"""),
    ("code", """from dotenv import load_dotenv
import anthropic
from evaluation_utils import calc_price, calc_total_price, llm_structured_retry, map_progress
load_dotenv()
client = anthropic.Anthropic()"""),
    ("code", """# Load the RAG answers to judge (the glm-5.2 answers we produced in notebook 3).
import pandas as pd, json
df_answers = pd.read_csv("data/rag-answers.csv")
answers = df_answers.to_dict(orient="records")
answers[0]"""),
    ("code", """# Judge ONE record.
rec = answers[0]
prompt = aqa_judge_prompt.format(question=rec["question"], answer_orig=rec["answer_orig"], answer_llm=rec["answer_llm"])
eval_result, usage = llm_structured_retry(client, aqa_judge_instructions, prompt, AnswerEvaluation)
print(eval_result.score, "|", eval_result.reasoning)
calc_price(usage)"""),
    ("code", """def evaluate_aqa(question, answer_orig, answer_llm):
    prompt = aqa_judge_prompt.format(question=question, answer_orig=answer_orig, answer_llm=answer_llm)
    return llm_structured_retry(client, aqa_judge_instructions, prompt, AnswerEvaluation)

def judge_record(rec):
    result, usage = evaluate_aqa(rec["question"], rec["answer_orig"], rec["answer_llm"])
    return ({
        "question": rec["question"], "document": rec["document"],
        "score": result.score, "reasoning": result.reasoning,
    }, usage)"""),
    ("md", """> **Cost control:** set `N_JUDGE`. The course judged all 395 (~$0.25). Here we
> judge all the glm-5.2 answers produced in notebook 3."""),
    ("code", """from concurrent.futures import ThreadPoolExecutor

N_JUDGE = len(answers)  # judge all answers from notebook 3
with ThreadPoolExecutor(max_workers=6) as pool:
    results = map_progress(pool, answers[:N_JUDGE], judge_record)

evaluations, usages = [], []
for ev, u in results:
    evaluations.append(ev); usages.append(u)

print("judge cost: $%.6f" % calc_total_price(usages))"""),
    ("code", """df_eval = pd.DataFrame(evaluations)
print(df_eval["score"].value_counts())
print()
print("good ratio: %.2f%%" % (100 * (df_eval["score"] == "good").mean()))
df_eval.to_csv("data/rag-evaluations.csv", index=False)
df_eval.head()"""),
    ("code", """# The 'bad' rows are the most useful part of evaluation -> inspect them.
df_eval[df_eval["score"] == "bad"].head()"""),
    ("md", """## Judging the judge

The judge itself can be wrong (e.g. too lenient). You can't audit it with *another*
judge — you must **read a sample of verdicts yourself** and disagree/agree, then
tighten the instructions and re-run. A Streamlit app showing Q / A / A' / verdict
side by side is the course's suggested workflow.

**Reference result (course, full 395 run):** 379 good / 16 bad (~96% good), ~$0.25.
For agents (lesson 14) the same idea applies, plus you save the **tool-call
trajectory** and judge answer quality + trajectory quality separately."""),
])


# ===========================================================================
# Write all four
# ===========================================================================
for fname, book in [
    ("01-data-gen.ipynb", data_gen),
    ("02-search-eval.ipynb", search_eval),
    ("03-rag-evals.ipynb", rag_evals),
    ("04-llm-judge.ipynb", llm_judge),
]:
    with open(fname, "w") as f:
        nbf.write(book, f)
    print("wrote", fname)
