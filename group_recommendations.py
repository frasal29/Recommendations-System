import pandas as pd
import csv
from pearson_similarity import pearson_similarity
from predict_rating import predict_rating_for_movie


def predict_ratings_for_unrated_movie(user_id, n, data, neighbors, correlation, movie_data):
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

     # Normalize predicted ratings
    for movie_id, predicted_rating in predictions.items():
        # Normalize using min-max scaling
        normalized_rating = 0.5 + ((predicted_rating - min_rating) / (max_rating - min_rating)) * 4.5
        predictions[movie_id] = normalized_rating
    
    return predictions


def generate_group_predictions(users, data, movie_data, neighbors):
    group_predictions = []

    for user_id in users:
        # Calculate Pearson correlations for the current user
        correlations = pearson_similarity(user_id, data)

        # Get rating predictions for the current user
        predictions = predict_ratings_for_unrated_movie(user_id, 1, data, neighbors, correlations, movie_data)

        # Add predictions for the single user to the group data structure
        for movie_id, predicted_rating in predictions.items():
            group_predictions.append({'userId': user_id, 'movieId': movie_id, 'rating': predicted_rating})

    return group_predictions


def append_ratings_from_table(data, predictions, users):
        for index, row in data.iterrows():
            user_id = int(row['userId'])
            movie_id = int(row['movieId'])
            rating = float(row['rating'])
            if user_id in users:
                predictions.append({'userId': user_id, 'movieId': movie_id, 'rating': rating})
        return predictions


def average_aggregation(file_path, users):
    # Load data from the CSV file
    data = pd.read_csv(file_path)
    
    # Create a set of all unique movieIds
    unique_movie_ids = set(data['movieId'])
    
    # Initialize a set to store movieIds that have ratings from all three users
    movie_ids_with_all_users = set()
    
    # Iterate through each movieId
    for movie_id in unique_movie_ids:
        # Check if all three users have rated this movie
        if all(data[(data['movieId'] == movie_id) & (data['userId'] == user)].shape[0] > 0 for user in users):
            # If yes, add it to the set of movieIds with ratings from all three users
            movie_ids_with_all_users.add(movie_id)

    # Dictionary to save the averages for each movieId
    average_ratings = {}
    
    # Calculate the average ratings for each movieId with ratings from all three users
    for movie_id in movie_ids_with_all_users:
        # Select ratings corresponding to the movieId
        ratings_for_movie = data[data['movieId'] == movie_id]['rating']
        # Calculate the average rating
        average_rating = ratings_for_movie.mean()
        # Save the average in the dictionary
        average_ratings[movie_id] = average_rating
    
    # Sort the dictionary based on the value of averageAggregation (in descending order)
    sorted_average_ratings = dict(sorted(average_ratings.items(), key=lambda item: item[1], reverse=True))
    
    # Write the results to the "average_aggregation.csv" file
    with open('average_aggregation.csv', 'w', newline='') as csvfile:
        fieldnames = ['movieId', 'averageAggregation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for movie_id, average_aggregation in sorted_average_ratings.items():
            writer.writerow({'movieId': movie_id, 'averageAggregation': average_aggregation})


def least_misery_aggregation(file_path, users):
    # Load data from the CSV file
    data = pd.read_csv(file_path)
    
    # Create a set of all unique movieIds
    unique_movie_ids = set(data['movieId'])
    
    # Initialize a set to store movieIds that have ratings from all three users
    movie_ids_with_all_users = set()
    
    # Iterate through each movieId
    for movie_id in unique_movie_ids:
        # Check if all three users have rated this movie
        if all(data[(data['movieId'] == movie_id) & (data['userId'] == user)].shape[0] > 0 for user in users):
            # If yes, add it to the set of movieIds with ratings from all three users
            movie_ids_with_all_users.add(movie_id)
              
    # Dictionary to save the minimum rating for each movieId
    least_misery_ratings = {}
    
    # Calculate the minimum rating for each movieId
    for movie_id in movie_ids_with_all_users:
        # Select ratings corresponding to the movieId
        ratings_for_movie = data[data['movieId'] == movie_id]['rating']
        # Calculate the minimum rating
        least_misery_rating = ratings_for_movie.min()
        # Save the minimum rating in the dictionary
        least_misery_ratings[movie_id] = least_misery_rating
    
        # Sort the dictionary based on the value of averageAggregation (in descending order)
    sorted_average_ratings = dict(sorted(least_misery_ratings.items(), key=lambda item: item[1], reverse=True))
    
    # Write the results to the "least_misery_aggregation.csv" file
    with open('least_misery_aggregation.csv', 'w', newline='') as csvfile:
        fieldnames = ['movieId', 'leastMiseryAggregation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for movie_id, least_misery_aggregation in sorted_average_ratings.items():
            writer.writerow({'movieId': movie_id, 'leastMiseryAggregation': least_misery_aggregation})


def print_top_10_aggregations(file_path, movie_data, aggregation_type):
    # Read the CSV file using Pandas
    data = pd.read_csv(file_path)
    
    # Select the first 10 rows
    top_10_records = data.head(10)
    
    # Print the header
    print(f"\nTop 10 recommendations for selected group of users with {aggregation_type}, in descending order:")
    
    # Print the first 10 records with corresponding movie titles
    for index, row in top_10_records.iterrows():
        movie_id = row['movieId']
        aggregation_value = row[aggregation_type]
        movie_title = movie_data[movie_id]['title']
        print(f"Movie: '{movie_title}', {'rating'}: {aggregation_value}")



def common_idMovie_in_group(file_path, users):
    # Load data from the CSV file
    data = pd.read_csv(file_path)
    
    # Initialize a set to store movieIds that have ratings from all three users
    movie_ids_with_all_users = set()
    
    # Find movieIds with ratings from all three users
    for movie_id in set(data['movieId']):
        if all(data[(data['movieId'] == movie_id) & (data['userId'] == user)].shape[0] > 0 for user in users):
            movie_ids_with_all_users.add(movie_id)
    
    # Filter the data to keep only records with movieIds present in all three users
    filtered_data = data[data['movieId'].isin(movie_ids_with_all_users)]
    
    # Create a dictionary to store ratings for each movieId
    movie_ratings = {movie_id: {} for movie_id in movie_ids_with_all_users}
    
    # Iterate through each movieId
    for movie_id in movie_ids_with_all_users:
        # Get ratings for each user for this movieId
        for user in users:
            rating = filtered_data[(filtered_data['userId'] == user) & (filtered_data['movieId'] == movie_id)]['rating'].values[0]
            movie_ratings[movie_id][f'rating_user{user}'] = rating
    
    # Create a new DataFrame from the dictionary
    new_data = pd.DataFrame(movie_ratings).transpose().reset_index()
    new_data.columns = ['movieId'] + [f'rating_user{user}' for user in users]
    
    # Sort the DataFrame by movieId in descending order
    new_data = new_data.sort_values(by='movieId', ascending=True)
    
    return new_data


def calculate_average_score(item_scores):
    # Calcola il punteggio medio degli item nel gruppo
    return sum(item_scores) / len(item_scores)

def calculate_min_score(item_scores):
    # Calcola il punteggio minimo degli item nel gruppo
    return min(item_scores)

def calculate_pairwise_disagreement(item_scores):
    # Calcola la media delle differenze di coppia per gli item nel gruppo
    num_members = len(item_scores)
    total_disagreement = sum(abs(item_scores[i] - item_scores[j]) for i in range(num_members) for j in range(i + 1, num_members))
    return (2 / (num_members * (num_members - 1))) * total_disagreement

def calculate_final_score(item_scores, score_function, w=0.4):
    # Calcola il punteggio F(dz,G) utilizzando la combinazione di media e disaccordo
    score = score_function(item_scores)
    dis_score = calculate_pairwise_disagreement(item_scores)
    return (1 - w) * score + w * dis_score

def calculate_scores_from_csv(data, score_function, w=0.2):
    # Initialize empty lists to store scores
    scores = []
    dis_scores = []
    final_scores = []
    
    # Iterate through each record in the table
    for index, row in data.iterrows():
        # Extract ratings from the row
        ratings = row[1:].tolist()
        
        # Calculate score using the chosen function
        score = score_function(ratings)
        scores.append(score)

        # Calculate pairwise disagreement for the ratings
        dis_score = calculate_pairwise_disagreement(ratings)
        dis_scores.append(dis_score)
        
        # Calculate final score using chosen function and pairwise disagreement
        final_score = calculate_final_score(ratings, score_function, w)
        final_scores.append(final_score)
    
    # Add scores to the DataFrame
    data['aggregation_score'] = scores
    data['pairwise_disagreement'] = dis_scores
    data['final_score'] = final_scores
    
    return data

