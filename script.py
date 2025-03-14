import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recommend_movies_by_overview(movie_title, top_n=10):
    # Loading and read data
    df_movies = pl.read_csv(
        "movies_metadata.csv",
        schema_overrides={
            "adult": pl.Utf8,
            "budget": pl.Utf8,
            "id": pl.Utf8
        },
        ignore_errors=True,
        infer_schema_length=10000
    )
    
    # Filtrar filmes sem overview
    df_movies = df_movies.filter(
        ~pl.col("overview").is_null() & 
        (pl.col("overview") != "")
    )
    
    # Selecionar apenas as colunas necessárias
    df_movies = df_movies.select(["id", "original_title", "overview"])
    
    # Criar uma lista de todos os títulos e overviews para facilitar a busca
    all_titles = df_movies.select("original_title").to_series().to_list()
    all_overviews = df_movies.select("overview").to_series().to_list()
    
    # Criar vetores TF-IDF a partir dos overviews
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(all_overviews)
    
    # Calcular similaridade do cosseno entre todos os filmes
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Encontrar índice do filme de entrada
    idx = -1
    
    # Criar uma lista de títulos em minúsculas para pesquisa
    lower_titles = [title.lower() for title in all_titles]
    
    # Buscar o filme por título exato
    if movie_title.lower() in lower_titles:
        idx = lower_titles.index(movie_title.lower())
        found_title = all_titles[idx]
        print(f"Encontrado: '{found_title}'")
    else:
        # Buscar filmes que contenham o título informado
        partial_matches = [(i, title) for i, title in enumerate(lower_titles) if movie_title.lower() in title]
        
        if partial_matches:
            idx = partial_matches[0][0]  # Pegar o primeiro match parcial
            found_title = all_titles[idx]
            print(f"Filme exato não encontrado. Usando '{found_title}' como referência.")
    
    if idx == -1:
        print(f"Filme '{movie_title}' não encontrado.")
        return pl.DataFrame({"original_title": ["Filme não encontrado"]})
    
    # Obter pontuações de similaridade para o filme
    sim_scores = [(i, cosine_sim[idx][i]) for i in range(len(cosine_sim[idx]))]
    
    # Sort movies by similarity (descending)
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top_n most similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:top_n+1]
    
    # Get movie indexes
    movie_indices = [i[0] for i in sim_scores]
    
    # Get recommended movie titles
    recommended_titles = [all_titles[i] for i in movie_indices]
    
    # Create an DataFrame with results
    recommendations = pl.DataFrame({"original_title": recommended_titles})
    
    return recommendations

def main():
    try:
        # simple interface
        movie_title = input("Digite o nome de um filme: ")
        recommendations = recommend_movies_by_overview(movie_title)
        
        print("\nRecommendations movies:")
        print(recommendations)
    except Exception as e:
        print(f"Erro ao executar o sistema de recomendação: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()