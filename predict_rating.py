
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
