from ingest import load_faq_data
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np
from sqlitesearch import VectorSearchIndex

model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
vs_index = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path="faq_vectors2.db"
)

vs_index.clear()

documents = load_faq_data()
texts = []

for doc in documents:
    text = doc["question"] + " " + doc["answer"]
    texts.append(text)

batch_size = 50
vectors = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    batch_vectors = model.encode(batch)
    vectors.extend(batch_vectors)

X = np.array(vectors)

vs_index.fit(X, documents)
vs_index.close()