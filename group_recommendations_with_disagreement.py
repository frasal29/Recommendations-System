''' FUNCTIONS FOR PART B OF GROUP OF RECOMMENDATIONS WITH DISAGREEMENT '''

import pandas as pd

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

def calculate_final_score(item_scores, score_function, w):
    # Calcola il punteggio F(dz,G) utilizzando la combinazione di media e disaccordo
    score = score_function(item_scores)
    dis_score = calculate_pairwise_disagreement(item_scores)
    return (1 - w) * score + w * dis_score

def calculate_scores_from_csv(data, score_function, w):
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

def print_top_10_films(data_with_scores, movie_data):
    # Sort the DataFrame by final_score in descending order
    sorted_data = data_with_scores.sort_values(by='final_score', ascending=False)
    
    # Print only the top 10 predictions with the corresponding title
    print("\nTop 10 recommendations of selected users considering disagreements, in descending order:")
    for i, (_, row) in enumerate(sorted_data.head(10).iterrows()):
        movie_id = row['movieId']
        movie_title = movie_data[movie_id]['title']
        final_score = row['final_score']
        print(f"{i+1}. Movie: '{movie_title}', Final Score: {final_score}")