import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np

def system_recommendation(input_user, metadata_movies, metadata_keywords, metadata_rating):
    """
    Recommendation system that searches for movies based on keywords using only Polars
    """

    input_user = input_user.lower()
    
    metadata_keywords = metadata_keywords.with_columns(pl.col("id").cast(pl.Utf8))
    
    def extract_keywords(keywords_str):
        try:
            keywords_list = json.loads(keywords_str.replace("'", "\""))
            return " ".join([item['name'] for item in keywords_list])
        except:
            return ""
    
    metadata_keywords = metadata_keywords.with_columns([
        pl.col("keywords").map_elements(extract_keywords, return_dtype=pl.Utf8).alias("keywords_text")
    ])
    
    df_movies_keywords = metadata_movies.join(
        metadata_keywords.select(["id", "keywords_text"]),
        left_on="id",
        right_on="id",
        how="left"
    )
    
    average_ratings = metadata_rating.group_by("movieId").agg(
        pl.col("rating").mean().alias("average_score"),
        pl.col("rating").count().alias("total_avaliacoes")
    )
    
    average_ratings = average_ratings.with_columns([
        pl.col("movieId").cast(pl.Utf8).alias("movieId")
    ])
    
    df_complete = df_movies_keywords.join(
        average_ratings,
        left_on="id",
        right_on="movieId",
        how="left"
    )
    
    filmes_correspondentes = df_complete.filter(
        pl.col("title").str.to_lowercase().str.contains(input_user)
    )
    
    if filmes_correspondentes.height == 0:
        return pl.DataFrame()
    
    all_ids = df_complete.select("id").to_series().to_list()
    
    all_keyword_texts = df_complete.select("keywords_text").fill_null("").to_series().to_list()
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(all_keyword_texts)
    
    filme_id = filmes_correspondentes.select("id").row(0)[0]
    filme_idx = all_ids.index(filme_id)
    
    cosine_sim = cosine_similarity(tfidf_matrix[filme_idx:filme_idx+1], tfidf_matrix).flatten()
    
    similarity_scores = list(enumerate(cosine_sim))
    
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:10]
    
    similar_index = [i[0] for i in similarity_scores]
    
    similar_ids = [all_ids[idx] for idx in similar_index]
    
    result = df_complete.filter(pl.col("id").is_in(similar_ids))
    
    id_to_score = {}
    for idx, score in similarity_scores:
        id_to_score[all_ids[idx]] = score
    
    result = result.with_columns([
        pl.col("id").map_elements(lambda x: id_to_score.get(x, 0.0), return_dtype=pl.Float64).alias("similaridade")
    ])
    
    result_sorted = result.sort("similaridade", descending=True)
    
    final_result = result_sorted.select([
        "title", "release_date", "average_score", "total_avaliacoes", 
        "overview", "keywords_text", "similaridade"
    ])
    
    return final_result

def main():
    metadata_movies = pl.read_csv(
        "movies_metadata.csv",
        schema_overrides={"adult": pl.Utf8, "budget": pl.Utf8, "id": pl.Utf8},
        ignore_errors=True,
        infer_schema_length=10000
    )
    
    metadata_keywords = pl.read_csv("keywords.csv", ignore_errors=True)
    metadata_rating = pl.read_csv("ratings.csv", ignore_errors=True)
    
    while True:
        input_user = input("\nType in the name of the movie or related term (or 'exit' to close): ")
        
        if input_user.lower() == 'exit':
            break
            
        results = system_recommendation(input_user, metadata_movies, metadata_keywords, metadata_rating)
        
        if results.height == 0:
            print("No movies found with that term. Try another search.")
        else: 
            for i, row in enumerate(results.head(5).rows(named=True)):
                print(f"\n{i+1}. {row['title']} ({row['release_date'][:4] if row['release_date'] else 'N/A'})")
                
                if row['average_score'] is not None:
                    print(f"   IMDB Score: {row['average_score']:.1f}/5.0")
                else:
                    print("   IMDB Score:  Not available")
                
if __name__ == "__main__":
    main()