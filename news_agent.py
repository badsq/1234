import base64
from typing import List, Dict

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseMessage
from langgraph.graph import StateGraph

API_KEY = "AIzaSyAduyr-YOpKb1CO9a4BonN04Z_H815uMXs"

genai.configure(api_key=API_KEY)

# Helper function to load image as GenAI part

def image_to_part(path: str):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return genai.types.Part.from_data(data, mime_type="image/jpeg")

# -------- LangChain chains --------

FILTER_PROMPT = PromptTemplate(
    template="""Отфильтруй новость. Если текст релевантен, ответь 'keep', иначе 'drop'.\n\nИстория: {history}\nТекущий пост: {post}\n""",
    input_variables=["history", "post"],
)

RANK_PROMPT = PromptTemplate(
    template="""Оцени важность новости по шкале от 0 до 10 и выведи только число.\n\nНовость: {post}\n""",
    input_variables=["post"],
)

COMPOSE_PROMPT = PromptTemplate(
    template="""Сформулируй краткий пост для телеграма на основе новости и истории канала.\n\nИстория: {history}\nНовость: {post}\n""",
    input_variables=["history", "post"],
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)
filter_chain = LLMChain(llm=llm, prompt=FILTER_PROMPT)
rank_chain = LLMChain(llm=llm, prompt=RANK_PROMPT)
compose_chain = LLMChain(llm=llm, prompt=COMPOSE_PROMPT)

# -------- LangGraph pipeline --------

class State(Dict):
    posts: List[Dict]
    history: List[str]
    queue: List[Dict]
    results: List[str]

def filter_node(state: State) -> Dict:
    filtered = []
    for post in state["posts"]:
        res = filter_chain.invoke({"history": "\n".join(state["history"]), "post": post["text"]})
        if "keep" in res["text"].lower():
            filtered.append(post)
    return {"queue": filtered}

def rank_node(state: State) -> Dict:
    scored = []
    for post in state["queue"]:
        res = rank_chain.invoke({"post": post["text"]})
        try:
            score = float(res["text"].strip())
        except ValueError:
            score = 0.0
        scored.append((score, post))
    scored.sort(key=lambda x: x[0], reverse=True)
    return {"queue": [p for _, p in scored]}

def compose_node(state: State) -> Dict:
    results = []
    for post in state["queue"]:
        res = compose_chain.invoke({"history": "\n".join(state["history"]), "post": post["text"]})
        results.append(res["text"].strip())
    return {"results": results}

graph = StateGraph(State)
graph.add_node("filter", filter_node)
graph.add_node("rank", rank_node)
graph.add_node("compose", compose_node)

graph.add_edge("filter", "rank")
graph.add_edge("rank", "compose")

graph.set_entry_point("filter")

graph.set_finish_point("compose")

pipeline = graph.compile()

def run(posts: List[Dict], history: List[str]):
    state = {"posts": posts, "history": history, "queue": [], "results": []}
    result = pipeline.invoke(state)
    return result["results"]

if __name__ == "__main__":
    sample_posts = [
        {"text": "Llama 4 Scout и Maverick от Meta"},
        {"text": "TAPNext: Новый подход"},
        {"text": "какой хороший день"},
    ]
    history = []
    print(run(sample_posts, history))
