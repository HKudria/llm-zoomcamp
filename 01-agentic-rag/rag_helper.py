from toyaikit.tools import Tools
from toyaikit.chat.interface import StdOutputInterface
from toyaikit.chat.runners import AnthropicMessagesRunner, DisplayingRunnerCallback

INSTRUCTIONS = '''
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up information, use the search function. 
Use as many keywords from the user question as possible when making first requests.

Make multiple searches. First perform search, analyze the results 
and then perform more searches. 

The question has to be about the course or its logistics, offtopic questions 
shouldn't be answered. If the search returns nothing, it's likely an off-topic question.
If you can't answer the question using FAQ, don't do it yourself. Only use the 
facts from the FAQ database.

At the end, ask if there are other areas that the user wants to explore.
'''.strip()

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
        model,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        # course='llm-zoomcamp',
        agent_tools=Tools(),
        chat_interface=StdOutputInterface(),
        callback = ''
    ):
        self.messageHistory = []
        self.iteration = 1
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        # self.course = course
        self.prompt_template = prompt_template
        self.model = model
        agent_tools.add_tool(self.search)
        self.callback = DisplayingRunnerCallback(chat_interface)
        self.runner = AnthropicMessagesRunner(
            tools=agent_tools,
            developer_prompt=instructions,
            chat_interface=chat_interface,
            llm_client=self.llm_client
        )

    def search(self, query: str) -> dict[str, str]:
        """
        Search the FAQ database for entries matching the given query.
        """
        boost_dict = {'content': 3.0, 'filename': 0.5}
        # filter_dict = {'course': self.course}

        return self.index.search(
            query,
            num_results=5,
            boost_dict=boost_dict,
            # filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['filename'])
            lines.append('A: ' + doc['content'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    search_tool = {
        "type": "function",
        "name": "search",
        "description": "Search the FAQ database for entries matching the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text to look up in the course FAQ."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }

    def llm(self, messages):
        print(f"Iteration - {self.iteration}")
        self.iteration = self.iteration + 1
        message = self.llm_client.messages.create(
            max_tokens=1024,
            system=self.instructions,
            messages=messages,
            model=self.model,
            tools=[self.search_tool]
        )
        print(message.usage.input_tokens)
        return message

    def loop(self, query):
        agent_response = ''
        messages = [{"role": "user", "content": query}]
        response = self.llm(messages)
        messages.append({"role": "assistant", "content": response.content})

        while True:
            has_function_call = False
            for block in response.content:
                if block.type == "tool_use" and block.name == "search":
                    args = block.input
                    print(args)
                    search_results = self.search(args["query"])
                    context = self.build_context(search_results)
                    tool_result = {
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": context
                        }]
                    }
                    messages.append(tool_result)
                    response = self.llm(messages)
                    messages.append({"role": "assistant", "content": response.content})
                    has_function_call = True

                elif block.type == "text":
                    agent_response = block.text

            if not has_function_call:
                break

        return agent_response




    def rag(self, query):
        response = self.loop(query)
        return response
