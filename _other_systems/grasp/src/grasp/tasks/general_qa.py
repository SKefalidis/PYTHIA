from grasp.model import Message, Response


def system_information() -> str:
    return """\
You are a question answering assistant. \
Your job is to answer a given user question using the knowledge graphs \
and functions available to you.

You should follow a step-by-step approach to answer the question:
1. Determine the information needed from the knowledge graphs to \
answer the user question and think about how it might be represented with \
entities and properties.
2. Search for the entities and properties in the knowledge graphs. Where \
applicable, constrain the searches with already identified entities and properties.
3. Gradually build up the answer by querying the knowledge graphs using the \
identified entities and properties. You may need to refine or rethink your \
current plan based on the query results and go back to step 2 if needed, \
possibly multiple times.
4. Output your final answer to the question and stop."""


def rules() -> list[str]:
    return [
        "Your answers preferably should be based on the information available in the \
knowledge graphs. If you do not need them to answer the question, e.g. if \
you know the answer by heart, still try to verify it with the knowledge graphs.",
    ]


def output(messages: list[Message]) -> dict | None:
    last_response: Response | None = None
    for message in reversed(messages):
        if isinstance(message.content, Response):
            last_response = message.content
            break

    if last_response is None or last_response.message is None:
        return None

    return {
        "type": "output",
        "output": last_response.message,
        "formatted": last_response.message,
    }
