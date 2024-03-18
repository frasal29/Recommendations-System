
import pandas as pd
def cosin_similarity(user_id, data):
    correlations = {}  # Dictionary to store correlations with other users
    
    # Extract movieIds of the fixed user
    user_movies = set(data[data['userId'] == user_id]['movieId'])
    
    # Iterate over all other users in the dataset
    for other_user_id in data['userId'].unique():
        # If the user ID is different from user_id and has at least one movieId in common with the fixed user, calculate the correlation
        if other_user_id != user_id:
            # Extract common movieIds between the fixed user and the other user
            common_movies = user_movies.intersection(set(data[data['userId'] == other_user_id]['movieId']))

            # If there are no common movieIds, move to the next user
            if not common_movies:
                continue

            # Initialize numerator and denominators
            numerator = 0
            denominator = 0
            sum_a = 0
            sum_b = 0

            # Iterate over each common movieId
            for movie_id in common_movies:
                # Extract ratings for this movieId
                user_rating = data[(data['userId'] == user_id) & (data['movieId'] == movie_id)]['rating'].iloc[0]
                other_user_rating = data[(data['userId'] == other_user_id) & (data['movieId'] == movie_id)]['rating'].iloc[0]
                
                # Update numerator
                numerator += user_rating * other_user_rating

                # Update denominators
                sum_a += user_rating** 2
                sum_b += other_user_rating** 2
    
            # Calculate the final denominator as square root of the product of denominators
            denominator = (sum_a ** 0.5) * (sum_b ** 0.5)

            # If the denominator is zero, set the correlation to 0 to avoid division by zero
            if denominator == 0:
                correlation = 0
            else:
                # Calculate cosine correlation
                correlation = numerator / denominator
            
            correlations[other_user_id] = correlation

    return correlations


def pearson_similarity(user_id, data):
    correlations = {}  # Dictionary to store correlations with other users
    
    # Extract movieIds of the fixed user
    user_movies = set(data[data['userId'] == user_id]['movieId'])
    
    # Iterate over all other users in the dataset
    for other_user_id in data['userId'].unique():
        # If the user ID is different from user_id and has at least one movieId in common with the fixed user, calculate the correlation
        if other_user_id != user_id:
            # Extract common movieIds between the fixed user and the other user
            common_movies = user_movies.intersection(set(data[data['userId'] == other_user_id]['movieId']))

            # If there are no common movieIds, move to the next user
            if not common_movies:
                continue

            # Initialize numerator and denominators
            numerator = 0
            user_denominator = 0
            other_user_denominator = 0

            # Iterate over each common movieId
            for movie_id in common_movies:
                # Extract ratings for this movieId
                user_rating = data[(data['userId'] == user_id) & (data['movieId'] == movie_id)]['rating'].iloc[0]
                other_user_rating = data[(data['userId'] == other_user_id) & (data['movieId'] == movie_id)]['rating'].iloc[0]
                
                # Calculate the mean ratings for the fixed user and the other user
                user_mean = data[data['userId'] == user_id]['rating'].mean()
                other_user_mean = data[data['userId'] == other_user_id]['rating'].mean()

                # Update numerator
                numerator += (user_rating - user_mean) * (other_user_rating - other_user_mean)

                # Update denominators
                user_denominator += (user_rating - user_mean) ** 2
                other_user_denominator += (other_user_rating - other_user_mean) ** 2
            
            # Calculate the final denominator as square root of the product of denominators
            denominator = (user_denominator ** 0.5) * (other_user_denominator ** 0.5)

            # If the denominator is zero, set the correlation to 0 to avoid division by zero
            if denominator == 0:
                correlation = 0
            else:
                # Calculate Pearson correlation
                correlation = numerator / denominator
            
            correlations[other_user_id] = correlation

    return correlations


def predict_rating_for_movie(user_id, movie_id, relevant_users, data):
    # Calculate the numerator and denominator for rating prediction
    numerator = 0
    denominator = 0
    
    for other_user_id, similarity in relevant_users.items():
        # Extract the rating of the current user for the current movie
        other_user_rating = data[(data['userId'] == other_user_id) & (data['movieId'] == movie_id)]['rating'].iloc[0]
        
        # Calculate the mean rating of the current user
        other_user_mean = data[data['userId'] == other_user_id]['rating'].mean()
        
        # Update numerator and denominator
        numerator += similarity * (other_user_rating - other_user_mean)
        denominator += similarity
    
    # Calculate the predicted rating
    user_mean = data[data['userId'] == user_id]['rating'].mean()
    predicted_rating = user_mean + (numerator / denominator) if denominator != 0 else 0
    
    return predicted_rating


def predict_ratings_for_unrated_movies(user_id, n, data, neighbors, correlation, movie_data):
    # Get the set of movies not rated by the user
    unrated_movies = set(data['movieId']) - set(data[data['userId'] == user_id]['movieId'])
    
    # Get the top 40 similar users
    top_k_users = sorted(correlation.items(), key=lambda x: x[1], reverse=True)[:neighbors]
    
    predictions = {}  # Dictionary to store predictions
    
    # Find the minimum and maximum predicted ratings
    min_rating = float('inf')
    max_rating = float('-inf')
    
    for movie_id in unrated_movies:
        # Get the users among the top 40 who have rated the current movie
        rated_by_top_k_users = [other_user_id for other_user_id, _ in top_k_users if movie_id in set(data[data['userId'] == other_user_id]['movieId'])]
        
        # If the number of users who have rated the movie is greater than or equal to n, calculate the rating prediction
        if len(rated_by_top_k_users) >= n:
            # Filter the top 40 users to include only those who have rated the current movie
            relevant_users = {other_user_id: correlation for other_user_id, correlation in top_k_users if movie_id in set(data[data['userId'] == other_user_id]['movieId'])}
            
            # Calculate the rating prediction for the current movie
            predicted_rating = predict_rating_for_movie(user_id, movie_id, relevant_users, data)
            
            # Update minimum and maximum predicted ratings
            min_rating = min(min_rating, predicted_rating)
            max_rating = max(max_rating, predicted_rating)
            
            # Add the prediction to the dictionary
            predictions[movie_id] = predicted_rating
            
            # Print the prediction for the current movie with the corresponding title
            #print(f"Rating prediction for the movie '{movie_data[movie_id]['title']}': {predicted_rating}")
    
    # Normalize predicted ratings
    for movie_id, predicted_rating in predictions.items():
        # Normalize using min-max scaling
        normalized_rating = 0.5 + ((predicted_rating - min_rating) / (max_rating - min_rating)) * 4.5
        predictions[movie_id] = normalized_rating
    
    # Print the top 10 predictions in descending order
    print_top_10_predictions(predictions, movie_data)

def print_top_10_predictions(predictions, movie_data):
    # Sort the predictions in descending order
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    # Print only the top 10 predictions with the corresponding title
    print("\nTop 10 predictions in descending order:")
    for i, (movie_id, rating) in enumerate(sorted_predictions[:10]):
        movie_title = movie_data[movie_id]['title']
        print(f"{i+1}. Movie: '{movie_title}', Rating Prediction: {rating}")



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

