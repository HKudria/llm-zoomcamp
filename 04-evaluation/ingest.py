"""
Data loading + search index builder.

This file is provider-agnostic (no LLM SDK). It downloads the course FAQ
from datatalks.club and builds a `minsearch` keyword+TF-IDF index over it.

We reuse the same helpers across the whole module: every notebook starts by
loading the documents and building the index with `build_index`.
"""

import requests
from minsearch import Index


def load_faq_data():
    """Download the full FAQ dataset (several courses) as a list of dicts."""
    docs_url = 'https://datatalks.club/faq/json/courses.json'
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix = 'https://datatalks.club/faq'

    for course in courses_raw:
        course_url = f'{url_prefix}{course["path"]}'
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()

        documents.extend(course_data)

    return documents


def build_index(documents):
    """Build a minsearch Index.

    `text_fields` are searched with TF-IDF (the more rare a matching term, the
    higher the score); `keyword_fields` are used for exact filtering (here, by
    course). Boosts (set later) let us weight one field more than another.
    """
    index = Index(
        text_fields=['question', 'section', 'answer'],
        keyword_fields=['course']
    )
    index.fit(documents)
    return index
