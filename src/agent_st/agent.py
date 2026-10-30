import os
import sqlite3
import requests
from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from agent_st.database import DB_FILE, run_sqlite_database
from agent_st.langfuse_client import get_prompt_from_langfuse

load_dotenv()

langfuse = get_client()
langfuse_handler = CallbackHandler()

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_API_BASE"),
)

model = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_API_BASE"),
    temperature=0,
)

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=os.getenv("QDRANT_COLLECTION_NAME", "Capstone_3"),
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    check_compatibility=False,
)

sql_db = run_sqlite_database()
sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=model)
sql_tools = sql_toolkit.get_tools()


def _query(sql: str, params: tuple = ()) -> list[dict]:
    """Jalankan SELECT terparameter ke movies.db dan kembalikan list dict."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _fmt(value, suffix: str = ""):
    """None/NaN jadi '-', angka bulat (1995.0) jadi 1995."""
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value}{suffix}"


def _movie_line(r: dict) -> str:
    return (
        f"Title: {_fmt(r['Series_Title'])} | Year: {_fmt(r['Released_Year'])} | "
        f"Genre: {_fmt(r['Genre'])} | Runtime: {_fmt(r['Runtime'], ' min')} | "
        f"Certificate: {_fmt(r['Certificate'])} | Rating: {_fmt(r['IMDB_Rating'])} | "
        f"Director: {_fmt(r['Director'])} | Star: {_fmt(r['Star1'])} | "
        f"Gross: {_fmt(r['Gross'])} | Overview: {_fmt(r['Overview'])}"
    )


class MovieSearchInput(BaseModel):
    keyword: str = Field(
        description="Movie title or part of a title to search for")


@tool("search_exact_movies_from_raw_dataset", args_schema=MovieSearchInput)
def search_exact_movies_from_raw_dataset(keyword: str) -> str:
    """Look up a specific named movie by title (or part of a title) in the
    movies database and return its details: year, genre, runtime, certificate,
    rating, director, main star, gross, overview. Shows at most 3 matches."""
    rows = _query(
        "SELECT * FROM movies "
        "WHERE instr(lower(Series_Title), lower(?)) > 0 "
        "ORDER BY IMDB_Rating DESC",
        (keyword,),
    )

    if not rows:
        return f"No movies found matching '{keyword}'."

    lines = [f"{len(rows)} match(es), showing {min(len(rows), 3)}."]
    lines += [_movie_line(r) for r in rows[:3]]
    return "\n".join(lines)


@tool
def get_relevant_movies(question: str) -> list:
    """Recommend movies by theme, plot, mood, or similarity to another movie
    (semantic search). It cannot filter by year, rating, or director; use the
    SQL tools for that."""
    docs = qdrant.as_retriever(search_kwargs={"k": 5}).invoke(question)

    if not docs:
        return []

    try:
        res = requests.post(
            "https://openrouter.ai/api/v1/rerank",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
            json={
                "model": "voyageai/rerank-2.5-lite",
                "query": question,
                "documents": [d.page_content for d in docs],
                "top_n": 3,
            },
            timeout=30,
        )
        res.raise_for_status()
        ranked = [docs[item["index"]]
                  for item in res.json().get("results", [])]

    except Exception as e:
        print(f"[rerank gagal] {e}")
        ranked = []

    return [
        f"Movie Title: {d.metadata.get('title')}\n"
        f"IMDb Rating: {d.metadata.get('imdb_rating')}\n"
        f"Genre: {d.metadata.get('genre')}\n"
        f"Year: {d.metadata.get('year')}\n"
        f"Synopsis: {d.page_content}"
        for d in ranked
    ]


class MovieComparisonInput(BaseModel):
    movie_1: str = Field(description="First movie title to compare")
    movie_2: str = Field(description="Second movie title to compare")


def _find_movie(title: str):
    """Cari satu film: utamakan judul persis, lalu judul yang mengandung kata kunci."""
    rows = _query(
        "SELECT * FROM movies WHERE lower(Series_Title) = lower(?)",
        (title.strip(),),
    )
    if not rows:
        rows = _query(
            "SELECT * FROM movies "
            "WHERE instr(lower(Series_Title), lower(?)) > 0 "
            "ORDER BY IMDB_Rating DESC",
            (title.strip(),),
        )
    return rows[0] if rows else None


@tool("compare_movies", args_schema=MovieComparisonInput)
def compare_movies(movie_1: str, movie_2: str) -> str:
    """Compare two named movies side by side: rating, gross, genre, year."""
    first, second = _find_movie(movie_1), _find_movie(movie_2)

    missing = [name for name, row in ((movie_1, first), (movie_2, second))
               if row is None]
    if missing:
        return "Could not find these movies in the dataset: " + ", ".join(missing)

    lines = []
    for r in (first, second):
        lines.append(
            f"Title: {_fmt(r['Series_Title'])} | Year: {_fmt(r['Released_Year'])} | "
            f"Rating: {_fmt(r['IMDB_Rating'])} | Gross: {_fmt(r['Gross'])} | "
            f"Genre: {_fmt(r['Genre'])}"
        )
    return "\n".join(lines)


tools = [search_exact_movies_from_raw_dataset,
         get_relevant_movies, compare_movies] + sql_tools

compiled_prompt = get_prompt_from_langfuse(
    "movie-agent-system-prompt", label="production")

agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=compiled_prompt,
)
