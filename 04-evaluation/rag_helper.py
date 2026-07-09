"""
Minimal RAG pipeline, adapted from the course to use the Anthropic SDK.

The course's `rag_helper.py` calls OpenAI's `responses.create` with a
`developer` message and reads `response.output_text`. With the Anthropic SDK
the equivalents are:
    - instructions  ->  the `system=` parameter
    - the prompt     ->  a `{"role": "user", "content": ...}` message
    - the answer     ->  `response.content[0].text`

`llm_client` is expected to be an `anthropic.Anthropic` instance. The model
defaults to the `ANTHROPIC_MODEL` env var (set to glm-5.2 in .env for z.ai).
"""

import os

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

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course='llm-zoomcamp',
        model=None,
        max_tokens=1024,
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        # fall back to the model configured for z.ai in .env
        self.model = model or os.getenv('ANTHROPIC_MODEL', 'glm-5.2')
        self.max_tokens = max_tokens

    def search(self, query, num_results=5):
        # boost the `question` field: a lexical match on the question is a
        # stronger signal than a match in the long `answer` text.
        boost_dict = {'question': 3.0, 'section': 0.5}
        filter_dict = {'course': self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['section'])
            lines.append('Q: ' + doc['question'])
            lines.append('A: ' + doc['answer'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        # Anthropic Messages API: system instructions live in `system=`,
        # the actual question goes in a user message.
        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.instructions,
            messages=[{'role': 'user', 'content': prompt}],
        )

        # The assistant reply is a list of content blocks; for a plain text
        # answer the first text block holds it.
        return response.content[0].text

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer
