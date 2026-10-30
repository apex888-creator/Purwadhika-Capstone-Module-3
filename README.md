# Movies Chatbot (Hercules)

AI chatbot for IMDb Top 1000 movies dataset, built with LangChain, LangGraph, Qdrant Cloud, SQLite, and Streamlit. Hercules answers questions about movies using only data returned by its tools, which prevents hallucinated titles, ratings, and figures.

## Features

- RAG recommendations: semantic search over movie titles, genres, and overviews in Qdrant Cloud, followed by a reranker that keeps the 3 most relevant results.
- Text-to-SQL: filtering, counting, averages, and rankings (for example "top 5 highest grossing movies") run against a SQLite database.
- Exact movie lookup and comparison: details and side-by-side comparison of named movies, read from SQLite with parameterized queries.
- Chat history: the last 20 messages are sent to the agent, so follow-ups such as "give me the details about the second one" work.
- Prompt management with Langfuse: the system prompt is versioned in Langfuse, and every run is traced.
- Transparency in the UI: expanders show the tool calls, the chat history (JSON), and LLM usage (input tokens, output tokens, and cost).
- Simple access check: a verification form with a limit on failed attempts before the chat opens.

## Movie Chatbot Flow

<img width="2006" height="1069" alt="image" src="https://github.com/user-attachments/assets/27f174cc-daa3-4f40-8592-ddbd188eacd1" />


## How It Works

The agent is a LangGraph ReAct agent. For each question it chooses which tool to call and answers only from the tool results.

| Tool | Data source | Used for |
| --- | --- | --- |
| `search_exact_movies_from_raw_dataset` | SQLite | Details of a specific named movie |
| `get_relevant_movies` | Qdrant Cloud + reranker | Recommendations by theme, plot, mood, or similarity |
| `compare_movies` | SQLite | Comparing two named movies |
| SQL tools (`SQLDatabaseToolkit`) | SQLite | Filters, counts, averages, rankings |

RAG flow in `get_relevant_movies`:

1. The question is embedded and Qdrant returns the 5 closest movies.
2. A reranker (`voyageai/rerank-2.5-lite`) scores the 5 candidates against the question.
3. The top 3 are returned to the agent as the tool result.

## Tech Stack

- Python, Poetry
- LangChain, LangGraph
- Langfuse (prompt management and tracing)
- Qdrant Cloud (vector database)
- SQLite (Text-to-SQL and exact lookup)
- OpenRouter (LLM, embedding model, and reranker)
- Streamlit

## Models

| Purpose | Model | Provider |
| --- | --- | --- |
| LLM (agent) | anthropic/claude-haiku-5.5 | OpenRouter |
| Embedding | text-embedding-3-small (1536 dimensions) | OpenAI via OpenRouter |
| Reranker | voyageai/rerank-2.5-lite | OpenRouter |

The LLM runs with temperature 0 for consistent answers. The embedding model must output 1536-dimensional vectors, because the Qdrant collection is created with that size.

## Dataset

[IMDb Top 1000](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows) stored at `dataset/imdb_top_1000.csv`. The dataset contains only the top 1000 titles, so a movie that is missing from it may still exist on IMDb.

## Project Structure

```
.
├── dataset/
│   ├── imdb_top_1000.csv
│   └── movies.db              # generated, not committed
├── src/
│   └── agent_st/
│       ├── agent.py           # tools and agent definition
│       ├── database.py        # Qdrant ingestion and SQLite creation
│       ├── langfuse_client.py # fetches the system prompt from Langfuse
│       └── simulation.py      # Streamlit app
├── tests/
├── pyproject.toml
└── README.md
```

## Local Installation

1. Clone this repository

   ```
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Install dependencies using Poetry:

   ```
   poetry install
   ```

3. Set up environment variables in `.env`:

   ```
   OPENROUTER_API_KEY=
   OPENROUTER_API_BASE=https://openrouter.ai/api/v1
   LLM_MODEL=
   EMBEDDING_MODEL=
   QDRANT_URL=
   QDRANT_API_KEY=
   QDRANT_COLLECTION_NAME=
   LANGFUSE_PUBLIC_KEY=
   LANGFUSE_SECRET_KEY=
   LANGFUSE_HOST=
   ```

   Never commit `.env` to GitHub.

4. Create the Langfuse prompt:

   In Langfuse, create a chat prompt named `movie-agent-system-prompt` with a single System message. `agent.py` loads it with the `latest` label. Change the label to `production` once the prompt is stable.

5. Prepare the data (once):

   ```
   poetry run python -m agent_st.database qdrant
   poetry run python -m agent_st.database
   ```

   The first command embeds the 1000 movies and uploads them to Qdrant Cloud (it uses the embedding API, so run it only when the collection is empty). The second command builds `dataset/movies.db` from the CSV.

## Running Locally

### Streamlit Simulation

```
poetry run streamlit run src/agent_st/simulation.py
```

## User Query Example

1. `give me details about The Godfather` --> calling "search_exact_movies_from_raw_dataset" tool.
2. `tell me about Avengers` --> calling "search_exact_movies_from_raw_dataset" tool but request back to user for clarity because multiple Avenger movie series in the dataset.
3. `I want a single heartwarming movie about friendship, what would you recommend ?` --> calling "get_relevant_movies" RAG tool.
4. `give me three movies similar to wolf of wall street and rank those based on highest IMDB ratings ?` --> calling "get_relevant_movies" RAG tool and implement reranker.
5. `give me the details about the second one` --> calling "search_exact_movies_from_raw_dataset" tool to extract information related to the previous LLM output.
6. `tell me about The Matrix, then suggest movies similar to it and rank those based on IMDB rating from highest to lowest` --> calling "search_exact_movies_from_raw_dataset" tool first and then calling "get_relevant_movies" RAG tool and implement reranker.
7. `compare fight club, the godfather and lethal weapon movies` --> calling "compare_movies" tool.
8. `what is the average IMDb rating of all Avengers movies?` --> calling "sql_tools" tool.
9. `what time now?` --> model refused to answer questions unrelated to the IMDB dataset.
10. `tell me the details about Godzilla Minus One` --> model refused to answer information cannot be found in the IMDB dataset.

## Deployment (Streamlit Community Cloud)

1. Push the repository to GitHub, including `dataset/imdb_top_1000.csv` (but not `.env` or `movies.db`).
2. Create a new app on Streamlit Community Cloud and set the main file to `src/agent_st/simulation.py`.
3. Add every variable from the `.env` list above under Settings, then Secrets.
4. Deploy. `movies.db` is created automatically the first time the app starts.

## Notes

- Answers come only from tool results. If a movie is not found, Hercules says so instead of guessing.
- SQL tools are instructed to run SELECT queries only. This is a prompt rule, not a database restriction.
- Usage cost in the UI is estimated from the token counts and the prices set in `simulation.py`.
