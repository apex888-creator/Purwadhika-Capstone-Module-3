from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_qdrant import QdrantVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from qdrant_client import QdrantClient
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import os
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

lf = Langfuse()
langfuse_handler = CallbackHandler()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    check_compatibility=False
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

model = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

# Memuat dataset IMDb Top 1000
df = pd.read_csv('dataset/imdb_top_1000.csv')


class MovieSearchInput(BaseModel):
    keyword: str = Field(
        description="Judul film atau kata kunci yang ingin dicari")


@tool("cari_film_imdb_lengkap", args_schema=MovieSearchInput, return_direct=True)
def cari_film_imdb_lengkap(keyword: str) -> str:
    """Tool untuk mencari informasi film secara mendetail berdasarkan dataset IMDb."""
    hasil = df[df['Series_Title'].str.contains(keyword, case=False, na=False)]

    if hasil.empty:
        return f"Maaf, film dengan kata kunci '{keyword}' tidak ditemukan di dalam dataset."

    response_list = []
    for _, row in hasil.head(3).iterrows():
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
            f"🎬 **{title}** ({year})\n"
            f"🏷️ Genre: {genre} | Durasi: {runtime} | Usia: {certificate}\n"
            f"⭐ IMDb Rating: {rating} | Sutradara: {director}\n"
            f"⭐ Pemeran Utama: {star1}\n"
            f"💰 Pendapatan (Gross): {gross}\n"
            f"📖 Sinopsis: {overview}\n"
            "-----------------------------------"
        )
        response_list.append(movie_info)

    return "\n".join(response_list)


@tool
def get_relevant_movies(question):
    """Use this tools for get relevant information about movies."""
    results = qdrant_client.similarity_search(
        question,
        k=5
    )
    return [d.page_content for d in results]


# 1. Membuat ReAct Agent menggunakan LangGraph
tools = [cari_film_imdb_lengkap, get_relevant_movies]
agent_executor = create_react_agent(model, tools, prompt=SystemMessage(
    content="You are a helpful assistant that can answer questions about movies and retrieve relevant information from the IMDb dataset. Use the provided tools to find detailed information about movies based on user queries. If you cannot find the answer in the dataset, respond with 'Maaf, saya tidak dapat menemukan informasi yang relevan.'"))
supervisor_agent = create_react_agent(model, tools)

# 2. Contoh cara menjalankan agent dengan callback Langfuse
if __name__ == "__main__":
    query = "Tolong carikan informasi lengkap mengenai film Chaplin"

    print(f"Query: {query}\n" + "="*40)

    # Menjalankan agent dengan menyertakan langfuse_handler pada config
    response = agent_executor.invoke(
        {"messages": [("user", query)]},
        config={"callbacks": [langfuse_handler]}
    )

    # Menampilkan pesan terakhir dari agent
    print("\nJawaban Agent:")
    print(response["messages"][-1].content)

# @tool
# def search_resume(query: str, k: int = 5) -> list[str]:
#     """Retrieve relevant resumes on the query."""

#     vector_store = QdrantVectorStore(
#         client=qdrant_client,
#         collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
#         embedding=embeddings,
#     )

#     docs = vector_store.similarity_search_with_score(query, k=k)
#     if docs:
#         formatted_results = []
#         for idx, result in enumerate(docs):
#             formatted_results.append(f"""
# Resume {idx + 1}:
# - ID: {result[0].metadata.get('row_index', 'N/A')}
# - Category: {result[0].metadata.get('category', 'N/A')}
# - Relevance Score: {result[1]:.3f}
# - Content Preview: {result[0].page_content[:300]}...
# """)

#         context = "\n".join(formatted_results)
#         return context
#     return "No relevant documents found."


# @tool
# def search_resume_skill(query: str, k: int = 5) -> list[str]:
#     """Retrieve relevant resumes on the query."""

#     vector_store = QdrantVectorStore(
#         client=qdrant_client,
#         collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
#         embedding=embeddings,
#     )

#     docs = vector_store.similarity_search_with_score(query, k=k)
#     if docs:
#         formatted_data = []
#         for idx, result in enumerate(docs):
#             formatted_data.append(f"""
# Resume {idx + 1} ({result[0].metadata.get('row_index', 'N/A')}):
# {result[0].page_content[:500]}...
# """)

#         context = "\n".join(formatted_data)
#         return context
#     return "No relevant documents found."


# lf_resume_search = lf.get_prompt("resume_search_agent").get_langchain_prompt()

# resume_search_agent = create_react_agent(
#     model=model,
#     tools=[search_resume],
#     system_prompt=lf_resume_search
# )

# lf_skill_analyze = lf.get_prompt("skill_analyze_agent").get_langchain_prompt()

# skill_analyze_agent = create_react_agent(
#     model=model,
#     tools=[search_resume_skill],
#     system_prompt=lf_skill_analyze
# )


# @tool(
#     args_schema=AgentInput
# )
# def resume_search(query: str, history: str) -> str:
#     """Tool to search resumes using the resume search agent.
#     Use this when the user wants to  find/search for specific candidates or resumes

#     query: "find HR managers", "search for candidates with X skill".
#     history: chat history summary
#     """
#     result = resume_search_agent.invoke({
#         "messages": [{"role": "user", "content": query + "history chat: " + history}]
#     }, config={"callbacks": [langfuse_handler]})
#     return result["messages"][-1].text


# @tool(
#     args_schema=AgentInput
# )
# def skill_analyze(query: str, history: str) -> str:
#     """Tool to Analyze skills, create comparisons, identify gaps
#     Use when: User asks about skills, wants analysis or comparisons

#     query: "what skills does", "compare skills", "skills gap analysis
#     history: chat history
#     """
#     result = skill_analyze_agent.invoke({
#         "messages": [{"role": "user", "content": query + "history chat: " + history}]
#     }, config={"callbacks": [langfuse_handler]})
#     return result["messages"][-1].text


# lf_supervisor = lf.get_prompt("supervisor_agent").get_langchain_prompt()
# supervisor_agent = create_react_agent(
#     model=model,
#     tools=[search_resume, search_resume_skill],
#     system_prompt=lf_supervisor
# )
