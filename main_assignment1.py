import pandas as pd
from pearson_similarity import pearson_similarity
from cosin_similarity import cosin_similarity
from predict_rating import predict_ratings_for_unrated_movies

def load_movie_data(file_path):
    # Load data from the CSV file
    movie_data = pd.read_csv(file_path)
    # Return data as a dictionary where keys are movie IDs
    return {row['movieId']: {'title': row['title'], 'genres': row['genres']} for _, row in movie_data.iterrows()}

def main():
    # Load the dataset
    data = pd.read_csv("dataset/ml-latest-small/ratings.csv")

    # Load movie data
    movie_data = load_movie_data("dataset/ml-latest-small/movies.csv")
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Calculate Pearson correlation between users")
        print("2. Calculate cosine correlation between users")
        print("3. Make rating predictions for users")

        choice = input("Choice (1/2/3, q to quit): ")
        
        if choice == '1':
            user_id = input("Enter the reference user ID: ")
            correlations = pearson_similarity(int(user_id), data)
            print("Top 10 most similar users:")
            top_10_users = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
            for other_user_id, correlation in top_10_users:
                print(f"User {other_user_id}: Correlation = {correlation}")
        
        elif choice == '2':
            user_id = input("Enter the reference user ID: ")
            correlations = cosin_similarity(int(user_id), data)
            print("Top 10 most similar users:")
            top_10_users = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
            for other_user_id, correlation in top_10_users:
                print(f"User {other_user_id}: Correlation = {correlation}")
        
        elif choice == '3':
            user_id = input("Enter the user ID: ")
            correlation_type = input("Which type of correlation do you want to use? (pearson/cosin): ")
            neighbors = int(input("Enter the number of neighbors: "))
            if correlation_type == 'pearson':
                correlations = pearson_similarity(int(user_id), data)
            elif correlation_type == 'cosin':
                correlations = cosin_similarity(int(user_id), data)
            else:
                print("Invalid correlation type!")
                continue
            
            predict_ratings_for_unrated_movies(int(user_id), 1, data, neighbors, correlations, movie_data)
        
        elif choice.lower() == 'q':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
