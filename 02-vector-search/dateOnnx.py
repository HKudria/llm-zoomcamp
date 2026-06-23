from ingest import load_faq_data
from tqdm.auto import tqdm
import numpy as np

documents = load_faq_data()

texts = [doc["question"] + " " + doc["answer"] for doc in documents]

batch_size = 50
X = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    batch_vectors = embed.encode_batch(batch)
    X.extend(batch_vectors)

X = np.array(X)