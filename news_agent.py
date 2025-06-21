import os
from dataclasses import dataclass, field
from typing import List, Dict

from google import genai

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END


def get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)


def run_llm(prompt: str) -> str:
    client = get_client()
    model = "gemini-2.5-flash"
    contents = [
        genai.types.Content(role="user", parts=[genai.types.Part.from_text(prompt)])
    ]
    generate_content_config = genai.types.GenerateContentConfig(
        thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
        response_mime_type="text/plain",
    )
    response = client.models.generate_content(
        model=model, contents=contents, config=generate_content_config
    )
    return response.text


@dataclass
class Post:
    author: str
    date: str
    text: str
    images: List[str] = field(default_factory=list)


@dataclass
class PipelineState:
    post: Post
    history: List[Post]
    keep: bool = False
    score: float = 0.0
    output: str | None = None


# Prompt templates
EVAL_PROMPT = ChatPromptTemplate.from_template(
    """You are a news curator. Determine whether the following post should be\
 published in the ML/AI Telegram channel. Check relevance, match to channel\
 topics and duplication against previous posts.\n\nPost:\n{text}\n\nPrevious posts:\n{history}\n\nReply ONLY with KEEP or DISCARD and a short reason."""
)

SCORE_PROMPT = ChatPromptTemplate.from_template(
    """Rate the importance of this post from 0 to 10 based on how relevant and\
 newsworthy it is for an audience interested in AI, ML, DL, LLMs, VLMs, SD, and related topics.\n\nPost:\n{text}\n\nReturn only a number between 0 and 10."""
)

WRITE_PROMPT = ChatPromptTemplate.from_template(
    """Compose a concise Telegram post in Russian summarizing the following news.\
Focus on key facts and avoid duplicating information already posted.\n\n{text}\n"""
)


def evaluate(state: PipelineState) -> PipelineState:
    history_text = "\n\n".join(p.text for p in state.history[-5:])  # last 5 posts
    prompt = EVAL_PROMPT.format(text=state.post.text, history=history_text)
    response = run_llm(prompt)
    state.keep = response.strip().upper().startswith("KEEP")
    return state


def score(state: PipelineState) -> PipelineState:
    prompt = SCORE_PROMPT.format(text=state.post.text)
    response = run_llm(prompt)
    try:
        state.score = float(response.strip())
    except ValueError:
        state.score = 0.0
    return state


def write(state: PipelineState) -> PipelineState:
    prompt = WRITE_PROMPT.format(text=state.post.text)
    state.output = run_llm(prompt)
    return state


# Build LangGraph
workflow = StateGraph(PipelineState)
workflow.add_node("evaluate", evaluate)
workflow.add_node("score_node", score)
workflow.add_node("write_node", write)
workflow.set_entry_point("evaluate")
workflow.add_edge("evaluate", "score_node")
workflow.add_edge("score_node", "write_node")
workflow.add_edge("write_node", END)
flow = workflow.compile()


# Demo posts
def demo_posts() -> List[Post]:
    return [
        Post(author="Meta", date="2024-05-01", text="""📢 Новые модели Llama 4 Scout и Llama 4 Maverick от Meta\n\nПредставлены Llama 4 Scout и Llama 4 Maverick – самые продвинутые AI модели на данный момент."""),
        Post(author="Google", date="2024-05-02", text="""🎉 Google AI: Прорыв в медицинской диагностике с Project AMIE\n\nКоманда Google AI и Google DeepMind представила результаты Project AMIE."""),
        Post(author="Researcher", date="2024-05-03", text="""💡TAPNext: Новый подход к отслеживанию объектов в видео\n\nПредставлена модель TAPNext, устанавливающая новый стандарт."""),
    ]


def main():
    history: List[Post] = []
    posts = demo_posts()
    for post in posts:
        state = PipelineState(post=post, history=history)
        result = flow.invoke(state)
        if result.keep:
            print(f"\nScore: {result.score}\nPost:\n{result.output}\n")
            history.append(post)
        else:
            print(f"\nDiscarded post from {post.author}\n")


if __name__ == "__main__":
    main()
