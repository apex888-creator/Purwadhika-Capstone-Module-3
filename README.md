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
