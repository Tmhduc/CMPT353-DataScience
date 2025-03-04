import numpy as np
import pandas as pd
import sys
import difflib

# This function reads in all the movie titles from the movie_list.txt
def load_movie_titles(movie_list_file):
    """Load the list of correctly assumed spelled movie titles into a Pandas DataFrame"""
    movie_list = open(movie_list_file, 'r', encoding='utf-8').readlines()
    movie_list_dict = {'Movie Title': [movie.strip() for movie in movie_list]}
    return pd.DataFrame(movie_list_dict)

# Loads movie ratings from the movie_ratings.csv (needs cleaning data)
def load_movie_ratings(ratings_file):
    """Load the movie ratings from CSV"""
    return pd.read_csv(ratings_file)

def matching_titles(ratingsdf, moviedf):
    """Match movie ratings to the correct titles using difflib"""
    ratingsdf = ratingsdf.copy()
    ratingsdf['matched_title'] = ratingsdf['title'].apply(
        lambda title1: difflib.get_close_matches(
            title1, moviedf['Movie Title'].tolist(), n=2, cutoff=0.8
        )[0] if difflib.get_close_matches(
            title1, moviedf['Movie Title'].tolist(), n=2, cutoff=0.8
        ) else None
    )
    return ratingsdf.dropna(subset=["matched_title"])

def compute_average_ratings(ratingsdf):
    """Compute the average ratings for each correctly matched movie"""
    ratingsdf = ratingsdf.copy()
    ratingsdf["rating"] = ratingsdf["rating"].astype(float)
    avg_rating = (ratingsdf.groupby('matched_title')['rating']
                  .mean()
                  .round(2)
                  .reset_index()
                  .sort_values('matched_title')
                  .rename(columns={'matched_title': 'Movie Title'}))
    return avg_rating

def save_output(avg_ratings, output_file):
    avg_ratings.to_csv(output_file, index=False)
    
def main():
    if len(sys.argv) != 4:
        print("Usage: python3 average_ratings.py movie_list.txt movie_ratings.csv output.csv")
        sys.exit(1)

    movie_list_file = load_movie_titles(sys.argv[1])

    ratings_file = load_movie_ratings(sys.argv[2])
    matched_ratings_df = matching_titles(ratings_file, movie_list_file)
    output_file = sys.argv[3]
    avg_ratings = compute_average_ratings(matched_ratings_df)
    save_output(avg_ratings, output_file)    
if __name__ == "__main__":
    main()
    