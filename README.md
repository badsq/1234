# News Agent Demo

This repository contains a prototype for a news pipeline agent that processes posts from x.com and prepares them for publishing to Telegram channels.

The agent uses [LangChain](https://github.com/langchain-ai/langchain), [LangGraph](https://github.com/langchain-ai/langgraph) and the Google Generative AI (Gemini) API.

## Requirements

```
pip install -r requirements.txt
```

Set the `GOOGLE_API_KEY` environment variable with your Gemini API key before running the agent.

## Usage

Run the demo with:

```
python news_agent.py
```

The script uses synthetic posts for demonstration purposes.
