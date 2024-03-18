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
