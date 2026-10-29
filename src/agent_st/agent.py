import os
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel, Field
import requests
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.prebuilt import create_react_agent
from agent_st.database import run_sqlite_database
from agent_st.langfuse_client import get_prompt_from_langfuse

load_dotenv()

langfuse = get_client()
langfuse_handler = CallbackHandler()

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_API_BASE")
)

model = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_API_BASE"),
    temperature=0)

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="Capstone_3",
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    check_compatibility=False
)

df = pd.read_csv('dataset/imdb_top_1000.csv')

sql_db = run_sqlite_database()
sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=model)
sql_tools = sql_toolkit.get_tools()


class MovieSearchInput(BaseModel):
    keyword: str = Field(
        description="Movie title or keyword for search")


@tool("search_exact_movies_from_raw_dataset", args_schema=MovieSearchInput, return_direct=True)
def search_exact_movies_from_raw_dataset(keyword: str) -> str:
    """Tool to search for detailed movie information based on the IMDb dataset CSV."""
    result = df[df['Series_Title'].str.contains(keyword, case=False, na=False)]

    if result.empty:
        return f"Sorry, no movies found matching the keyword '{keyword}' in the IMDB Dataset CSV."

    response_list = []
    for _, row in result.head(3).iterrows():
        title = row.get('Series_Title', '-')
        year = row.get('Released_Year', '-')
        certificate = row.get('Certificate', '-')
        runtime = row.get('Runtime', '-')
        genre = row.get('Genre', '-')
        rating = row.get('IMDB_Rating', '-')
        director = row.get('Director', '-')
        star1 = row.get('Star1', '-')
        gross = row.get('Gross', '-')
        overview = row.get('Overview', '-')

        movie_info = (
            f"🎬 {title} ({year})\n"
            f"🏷️ Genre: {genre} | Duration: {runtime} | Certificate: {certificate}\n"
            f"⭐ IMDb Rating: {rating} | Director: {director}\n"
            f"⭐ Main Star: {star1}\n"
            f"💰 Gross Revenue: {gross}\n"
            f"📖 Overview: {overview}\n"
            "-----------------------------------"
        )
        response_list.append(movie_info)

    return "\n".join(response_list)


@tool
def get_relevant_movies(question: str) -> list:
    """Use this tools for get movies recommendations from IMDB dataset."""
    docs = qdrant.as_retriever(search_kwargs={"k": 5}).invoke(question)

    if not docs:
        return []

    res = requests.post(
        "https://openrouter.ai/api/v1/rerank",
        headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
        json={
            "model": "voyageai/rerank-2.5-lite",
            "query": question,
            "documents": [d.page_content for d in docs],
            "top_n": 3
        }
    ).json()

    ranked = [docs[item["index"]]
              for item in res.get("results", [])] or docs[:3]

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


@tool("compare_movies", args_schema=MovieComparisonInput)
def compare_movies(movie_1: str, movie_2: str) -> str:
    """Tool to compare two movies side by side based on IMDb dataset."""
    subset = df[df['Series_Title'].str.contains(
        f"{movie_1}|{movie_2}", case=False, na=False)]

    if len(subset) < 2:
        return "Could not find one or both movies for comparison."
    comparison_text = ""
    for _, row in subset.iterrows():
        title = row.get('Series_Title', '-')
        year = row.get('Released_Year', '-')

        comparison_text += (
            f"🎬 **{title}** ({year})\n"
            f"⭐ Rating: {row.get('IMDB_Rating')} | 💰 Gross: {row.get('Gross')} | 🏷️ Genre: {row.get('Genre')}\n"
            "----------------------------\n"
        )
    return comparison_text


tools = [search_exact_movies_from_raw_dataset,
         get_relevant_movies, compare_movies] + sql_tools
compiled_prompt = get_prompt_from_langfuse(
    "movie-agent-system-prompt", label="latest")

agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=compiled_prompt[0] if isinstance(
        compiled_prompt, list) else compiled_prompt,

)
# HAPUS AJA IF NAME MAINNYA
# TAMBAHIN KE SQLITE DI DATABASE.PY
# TAMBAHIN CONVERSATION HISTORY DAN MEMORY AGENT
if __name__ == "__main__":
    query = "Give me recommendation about action movie with high rating"

    print(f"Query: {query}\n" + "="*40)

    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"callbacks": [langfuse_handler]}
    )

    print("\nJawaban Agent:")
    print(response["messages"][-1].content)
