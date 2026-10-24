import os
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_qdrant import QdrantVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from qdrant_client import QdrantClient
from langfuse import get_client
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse = get_client()
langfuse_handler = CallbackHandler()

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

model = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1")

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="Capstone_3",
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    check_compatibility=False
)

df = pd.read_csv('dataset/imdb_top_1000.csv')


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
            f"🎬 **{title}** ({year})\n"
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
    results = qdrant.similarity_search(
        question,
        k=5
    )
    return [
        f"Title: {d.metadata.get('title')}\nSinopsis: {d.page_content}"
        for d in results
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
        comparison_text += (
            f"🎬 **{row.get('Series_Title')}** ({row.get('Released_Year')})\n"
            f"⭐ Rating: {row.get('IMDB_Rating')} | 💰 Gross: {row.get('Gross')} | 🏷️ Genre: {row.get('Genre')}\n"
            "-----------------------------------\n"
        )
    return comparison_text


tools = [search_exact_movies_from_raw_dataset,
         get_relevant_movies, compare_movies]

agent = create_react_agent(
    model=model,
    tools=[search_exact_movies_from_raw_dataset,
           get_relevant_movies, compare_movies],
    prompt="""You are an expert, precise, and factual AI assistant for an IMDb movie dataset application. Your primary goal is to provide accurate information strictly based on the provided tools and context.
                CRITICAL INSTRUCTIONS TO PREVENT HALLUCINATIONS:
                1. STRICT RELIANCE: Answer the user's question using ONLY the factual data returned by your tools or the provided context.
                2. NO ASSUMPTIONS: Do not extrapolate, guess, or invent details, ratings, titles, release years, or overviews that are not explicitly present in the tool results.
                3. ADMIT UNCERTAINTY: If the requested movie or information cannot be found in the dataset or tool outputs, explicitly state: "Sorry, I cannot find this information in the IMDb dataset." Do not make up answers.
                4. TOOL USAGE: Always utilize the available tools to fetch data before answering factual questions about specific movies.
                5. If the user asks for a specific piece of information (like year, rating, or gross revenue), answer concisely with just that specific data rather than showing the full movie profile.
                6. Never use markdown bold formatting (asterisks/stars) in your final answers. Output plain text only
                7. Do not use heading markdown symbols like # or ## for movie titles. Output them in regular text or bold text instead."""
)


# 2. Contoh cara menjalankan agent dengan callback Langfuse
if __name__ == "__main__":
    query = "Give me details about The Wolf of Wall Street"

    print(f"Query: {query}\n" + "="*40)

    # Menjalankan agent dengan menyertakan langfuse_handler pada config
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"callbacks": [langfuse_handler]}
    )

    # Menampilkan pesan terakhir dari agent
    print("\nJawaban Agent:")
    print(response["messages"][-1].content)
