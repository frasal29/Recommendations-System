# Group Movie Recommendations

This repository contains Python scripts for generating movie recommendations for single users and groups of users based on movie ratings. The recommendation system utilizes collaborative filtering techniques to predict ratings for unrated movies. Additionally, it offers the option to consider disagreement among group members when generating recommendations for the group.

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Usage](#usage)
   - [Single User Recommendations](#single-user-recommendations)
   - [Group Recommendations with or without Disagreement](#group-recommendations-with-or-without-disagreement)
4. [Scripts and Functions](#scripts-and-functions)
5. [License](#license)

## Introduction
This project aims to provide movie recommendations to users based on their movie ratings. It employs collaborative filtering algorithms such as Pearson correlation and cosine similarity to predict ratings for movies that a user has not yet rated. Furthermore, it offers the functionality to generate recommendations for groups of users, considering both their individual ratings and potential disagreements among them.

## Setup
To run the scripts in this repository, follow these steps:

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your_username/group-movie-recommendations.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the movie dataset (e.g., [MovieLens](https://grouplens.org/datasets/movielens/)) and place the CSV files in the `dataset/ml-latest-small/` directory.

## Usage
### Single User Recommendations
To generate movie recommendations for a single user:

```bash
python main_assignment1.py
```

Follow the prompts to choose the type of correlation and specify the user ID and the number of neighbors for the recommendation.


### Group Recommendations with or without Disagreement

To generate movie recommendations for a group of users, you can choose whether to consider disagreement among the users or not:

```bash
python main_assignment2.py
```

Within the code, you will be prompted to specify whether you want to generate predictions while considering disagreement.

Follow the prompts to enter the user IDs of the group members and choose the scoring function for aggregation.

## Scripts and Functions
- `main_assignment1.py`: Script for generating recommendations for a single user.
- `main_assignment2.py`: Script for generating recommendations for a group of users, considering or not disagreement among users
- `pearson_similarity.py`: Functions for calculating Pearson correlation between users.
- `cosin_similarity.py`: Functions for calculating cosine similarity between users.
- `predict_rating.py`: Functions for predicting ratings for movies.
- `group_recommendations.py`: Functions for generating recommendations in a group recommendations.
- `group_recommendations_with_disagreement.py`: Functions for generating recommendations in group recommendations considering disagreement.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

For any questions or feedback, feel free to contact [your_email@example.com](mailto:your_email@example.com).
