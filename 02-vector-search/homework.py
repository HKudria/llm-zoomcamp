from embedder import Embedder
from gitsource import GithubRepositoryDataReader, chunk_documents
import numpy as np
from minsearch import VectorSearch
from minsearch import Index

#Q1
embed = Embedder()
v = embed.encode('How does approximate nearest neighbor search work?')

print('v[0]:', v[0]) #-0.02

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = [file.parse() for file in reader.read()]

#Q2
for document in documents:
    if document['filename'] == '02-vector-search/lessons/07-sqlitesearch-vector.md':
        con = embed.encode(document['content'])
        print(con.dot(v)) #0.36107027225589694
        break

#Q3
chunks = chunk_documents(documents, size=2000, step=1000)

embedded_chunks=[]

for chunk in chunks:
    embedded_chunks.append(embed.encode(chunk['content']))

X = np.array(embedded_chunks)

scores = X.dot(v)
print(chunks[int(scores.argmax())]['filename']) #02-vector-search/lessons/07-sqlitesearch-vector.md

#Q4
vindex = VectorSearch(keyword_fields=["filename"])
vindex.fit(X, chunks)

results = vindex.search(embed.encode('What metric do we use to evaluate a search engine?'))
print(results[0]['filename']) #04-evaluation/lessons/05-search-metrics.md

#Q5
query = 'How do I store vectors in PostgreSQL?'

vector_results = vindex.search(embed.encode(query), num_results=5)

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(chunks)
text_results = index.search(query, num_results=5)
print('VECTOR')
for vector_result in vector_results:
    print(vector_result['filename'])

print()
print('INDEX')
for text_result in text_results:
    print(text_result['filename'])
#02-vector-search/lessons/08-pgvector.md

#Q6
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

query = 'How do I give the model access to tools?'

vector_results = vindex.search(embed.encode(query), num_results=5)

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(chunks)
text_results = index.search(query, num_results=5)

results = rrf([vector_results, text_results])
print()
print(results[0]['filename']) #01-agentic-rag/lessons/13-function-calling.md