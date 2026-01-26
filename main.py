import pandas as pd

def main():
    print("="*80)
    print("MOVIE RECOMMENDER DATA ANALYSIS")
    print("="*80)

    # =========================================================================
    # PART 1: Read and analyze ratings data (u.data)
    # =========================================================================
    print("\n" + "="*80)
    print("PART 1: RATINGS DATA ANALYSIS")
    print("="*80)

    # Read the u.data file
    # The file is tab-separated with columns: user_id, item_id, rating, timestamp
    ratings_df = pd.read_csv('data/u.data',
                             sep='\t',
                             names=['user_id', 'item_id', 'rating', 'timestamp'],
                             engine='python')

    # Display basic information about the dataset
    print(f"\nDataset shape: {ratings_df.shape}")
    print("\nFirst few rows:")
    print(ratings_df.head())
    print("\nDataset info:")
    print(ratings_df.info())
    print("\nBasic statistics:")
    print(ratings_df.describe())

    # =========================================================================
    # PART 2: Create binary multi-hot vectors for movies using genres
    # =========================================================================
    print("\n" + "="*80)
    print("PART 2: MOVIE GENRE VECTORS (MULTI-HOT ENCODING)")
    print("="*80)

    # Read genre information
    genres_df = pd.read_csv('data/u.genre', sep='|', names=['genre', 'genre_id'], encoding='latin-1')
    # Remove empty rows
    genres_df = genres_df.dropna()
    genre_names = genres_df['genre'].tolist()

    print(f"\nTotal genres: {len(genre_names)}")
    print(f"Genres: {genre_names}\n")

    # Read movie information
    # Columns: movie_id | movie_title | release_date | video_release_date | IMDb_URL | 19 genre binary columns
    column_names = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url'] + genre_names

    movies_df = pd.read_csv('data/u.item', sep='|', names=column_names, encoding='latin-1')

    print(f"Total movies: {len(movies_df)}")
    print("\nFirst few movies with their genre vectors:")
    print(movies_df.head())

    # Extract just the movie ID, title, and genre vectors
    genre_vectors = movies_df[['movie_id', 'movie_title'] + genre_names]

    print("\n" + "="*80)
    print("Movie Genre Vectors (Multi-Hot Encoding)")
    print("="*80)
    print(genre_vectors.head(10))

    # Show some statistics
    print("\n" + "="*80)
    print("Genre Statistics")
    print("="*80)
    genre_counts = movies_df[genre_names].sum().sort_values(ascending=False)
    print("\nNumber of movies per genre:")
    print(genre_counts)

    # Calculate average number of genres per movie
    avg_genres_per_movie = movies_df[genre_names].sum(axis=1).mean()
    print(f"\nAverage number of genres per movie: {avg_genres_per_movie:.2f}")

    # Save the genre vectors to a CSV file
    genre_vectors.to_csv('movie_genre_vectors.csv', index=False)
    print("\nGenre vectors saved to 'movie_genre_vectors.csv'")

    # Example: Show movies from a specific genre
    print("\n" + "="*80)
    print("Example: Movies with 'Sci-Fi' genre")
    print("="*80)
    scifi_movies = movies_df[movies_df['Sci-Fi'] == 1][['movie_id', 'movie_title'] + genre_names].head(10)
    print(scifi_movies)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
