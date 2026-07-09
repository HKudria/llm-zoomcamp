import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm.auto import tqdm

import anthropic

# make homework 2's embedder importable + locate its ONNX model
HW2 = Path(__file__).resolve().parent.parent / "02-vector-search"
sys.path.insert(0, str(HW2))
from embedder import Embedder  # noqa: E402

from gitsource import GithubRepositoryDataReader, chunk_documents  # noqa: E402
from minsearch import Index, VectorSearch  # noqa: E402

# the module's structured-output helper (Anthropic tool-forcing)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation_utils import llm_structured  # noqa: E402

load_dotenv()
client = anthropic.Anthropic()
MODEL_DIR = HW2 / "models/Xenova/all-MiniLM-L6-v2"


class Questions(BaseModel):
    questions: list[str]


data_gen_instructions = """
You emulate a student who is taking our LLM course.
You are given one lesson page from the course.
Formulate 5 questions this student might ask that are answered by this page.

Rules:
- The page should contain the answer to each question.
- Make the questions complete and not too short.
- Use as few words as possible from the page; don't copy its phrasing.
- The questions should resemble how people actually ask things online:
  not too formal, not too short, not too long.
- Ask about the content of the lesson, not about its formatting or filename.
""".strip()


reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]
print("lesson pages:", len(documents))

first3 = [
    "01-agentic-rag/lessons/01-intro.md",
    "01-agentic-rag/lessons/02-environment.md",
    "01-agentic-rag/lessons/03-rag.md",
]
doc_by_fn = {d["filename"]: d for d in documents}

input_tokens = []
for fn in first3:
    doc = doc_by_fn[fn]
    user_prompt = json.dumps({"filename": doc["filename"], "content": doc["content"]})
    result, usage = llm_structured(client, data_gen_instructions, user_prompt, Questions)
    input_tokens.append(usage.input_tokens)
    print(f"  {fn}: {len(result.questions)} q | in={usage.input_tokens} out={usage.output_tokens}")

avg_in = sum(input_tokens) / len(input_tokens)
options = [140, 1400, 14000, 140000]
closest = min(options, key=lambda o: abs(o - avg_in))
print(f"\nQ1  avg input tokens = {avg_in:.0f}  ->  closest option = {closest}")


chunks = chunk_documents(documents, size=2000, step=1000)
print("\nchunks:", len(chunks))

embed = Embedder(path=str(MODEL_DIR))
X = np.array(embed.encode_batch([c["content"] for c in chunks]))

vindex = VectorSearch(keyword_fields=["filename"])
vindex.fit(X, chunks)

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(chunks)


def vector_search(query, num_results=5):
    return vindex.search(embed.encode(query), num_results=num_results)


def text_search(query, num_results=5):
    return index.search(query, num_results=num_results)


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def hybrid_search(query, k=60):
    text_results = text_search(query, num_results=10)
    vector_results = vector_search(query, num_results=10)
    return rrf([text_results, vector_results], k=k)


df_gt = pd.read_csv("data/lessons-ground-truth.csv")
ground_truth = df_gt.to_dict(orient="records")
print("ground-truth questions:", len(ground_truth))


def compute_relevance(q, search_function):
    fn = q["filename"]
    results = search_function(query=q["question"])
    return [int(d["filename"] == fn) for d in results]


def compute_relevance_total(ground_truth, search_function):
    return [compute_relevance(q, search_function) for q in tqdm(ground_truth)]


def hit_rate(relevance):
    return sum(1 for line in relevance if 1 in line) / len(relevance)


def mrr(relevance):
    total = 0.0
    for line in relevance:
        for rank in range(len(line)):
            if line[rank] == 1:
                total += 1 / (rank + 1)
                break
    return total / len(relevance)


def evaluate(ground_truth, search_function):
    rel = compute_relevance_total(ground_truth, search_function)
    return {"hit_rate": hit_rate(rel), "mrr": mrr(rel)}

q0 = ground_truth[0]["question"]
print("\nQ0 question:", q0)
print("Q2  text_search   first:", text_search(q0)[0]["filename"])
print("Q3  vector_search first:", vector_search(q0)[0]["filename"])
print("    (generated from: 01-agentic-rag/lessons/01-intro.md)")

text_metrics = evaluate(ground_truth, text_search)
print(f"\nQ4  text_search   -> hit_rate={text_metrics['hit_rate']:.4f}  mrr={text_metrics['mrr']:.4f}")

vector_metrics = evaluate(ground_truth, vector_search)
print(f"Q5  vector_search -> hit_rate={vector_metrics['hit_rate']:.4f}  mrr={vector_metrics['mrr']:.4f}")


print("\nQ6  hybrid_search MRR by k:")
best_k, best_mrr = None, -1.0
for k in [1, 50, 100, 200]:
    res = evaluate(ground_truth, lambda query, k=k: hybrid_search(query, k=k))
    star = ""
    if res["mrr"] > best_mrr:
        best_mrr, best_k = res["mrr"], k
        star = "  <== best"
    print(f"   k={k:>3}: hit_rate={res['hit_rate']:.4f}  mrr={res['mrr']:.4f}{star}")
print(f"Q6  best k = {best_k}  (mrr={best_mrr:.4f})")
