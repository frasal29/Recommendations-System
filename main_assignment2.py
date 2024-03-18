import pandas as pd
import group_recommendations as group
import group_recommendations_with_disagreement as group_dis

def load_movie_data(file_path):
    # Load data from the CSV file
    movie_data = pd.read_csv(file_path)
    # Return data as a dictionary where keys are movie IDs
    return {row['movieId']: {'title': row['title'], 'genres': row['genres']} for _, row in movie_data.iterrows()}

def main():
    
    users = []
    # Load the data
    data = pd.read_csv("dataset/ml-latest-small/ratings.csv")
    movie_data = load_movie_data("dataset/ml-latest-small/movies.csv")
       
    while True:
        print("Enter the three desired users (from 1 to 610) separated by comma (e.g., 1,2,3):")
        user_input = input()
        users = [int(user_id.strip()) for user_id in user_input.split(',')]
        
        if len(users) != 3:
            print("Make sure to enter exactly 3 users. Please try again.")
        else:
            break

    while True:
        print("Enter the number of most similar neighbors to compare for each user (from 5 to 50):")
        neighbors = input()
        
        try:
            neighbors = int(neighbors)
            if neighbors < 5 or neighbors > 50:
                print("Make sure to enter a number between 5 and 50. Please try again.")
            else:
                break
        except ValueError:
            print("Make sure to enter an integer. Please try again.")

    # Generate predictions for the users
    print("Generating predictions for the selected users")
    predictions = group.generate_group_predictions(users, data, movie_data, neighbors)
    
    print("Merging the dataset with predictions")
    # Add ratings from the provided table to predictions
    updated_predictions = group.append_ratings_from_table(data, predictions, users)

    # Sort predictions by user_id
    updated_predictions.sort(key=lambda x: x['userId'])

    # Write predictions to a CSV file
    predictions_df = pd.DataFrame(updated_predictions)
    predictions_df.to_csv('dataset_with_also_predictions.csv', index=False)

    # Execute aggregation functions
    group.average_aggregation("dataset_with_also_predictions.csv", users)
    group.least_misery_aggregation("dataset_with_also_predictions.csv", users)

    # Print the top 10 records from the aggregation tables
    group.print_top_10_aggregations("average_aggregation.csv", movie_data, "averageAggregation")
    group.print_top_10_aggregations("least_misery_aggregation.csv", movie_data, "leastMiseryAggregation")
    
    

    ''' PART 2 WITH DISAGREEMENT '''

    file_path = 'dataset_with_also_predictions.csv'
    # Ask the user if they want to generate recommendations with disagreement or exit the program
    while True:
        choice = input("\n\nDo you want to generate recommendations taking into account the disagreement between users?\nEnter 'yes' to generate recommendations, 'no' to exit (yes/no): ").strip().lower()
        if choice == 'yes':
            # If the user wants to consider disagreement, continue
            break
        elif choice == 'no':
            # If the user does not want to consider disagreement, exit the program
            print("Goodbye!")
            return
        else:
            print("Invalid choice. Please respond with 'yes' or 'no'.")

    # Dataset with only movies common to both the dataset and the recommended ones among the selected users
    data = group_dis.common_idMovie_in_group(file_path, users)
    
    # Choose the scoring function by the user
    while True:
        choice = input("Choose the scoring function to use ('average' or 'leastMisery'): ").strip().lower()
        if choice == 'average' or choice == 'leastmisery':
            if choice == 'average':
                score_function = group_dis.calculate_average_score
            else:
                score_function = group_dis.calculate_min_score
            break
        else:
            print("Invalid choice. Please choose between 'average' or 'leastMisery'.")

    # Calculate scores for the data
    w = 0.3
    data_with_scores = group_dis.calculate_scores_from_csv(data, score_function, w)
    
    # Sort the DataFrame based on the final score in descending order
    data_with_scores = data_with_scores.sort_values(by='final_score', ascending=False)
    
    # Save the output to a new CSV file
    output_file_path = 'scores_with_disagreement.csv'
    data_with_scores.to_csv(output_file_path, index=False)
    print(f"Output saved to: {output_file_path}")
    
    # Print the top 10 values
    print("Top 10 values:")
    print(data_with_scores.head(10))

    # Print the top 10 films
    group_dis.print_top_10_films(data_with_scores, movie_data)


if __name__ == "__main__":
    main()
