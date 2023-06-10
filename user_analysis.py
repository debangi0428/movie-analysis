import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

### from scratch

users_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_csv('u.user', sep='|', names=users_cols)

# Load movie data
movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('u.item', sep='|', names=movies_cols, encoding='latin-1')

# Load ratings data
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('u.data', sep='\t', names=ratings_cols)

print("Users:", users.shape)
print("Movies:", movies.shape)
print("Ratings:", ratings.shape)

# Merge ratings with user and movie data
data = pd.merge(pd.merge(ratings, users), movies)

# Check for missing values
print(data.isnull().sum())
#IMDB_URL is missing for 13
# release date is missing for 9

# Drop unnecessary columns
data = data.drop(['video_release_date', 'IMDb_URL', 'unknown'], axis=1)

# Convert timestamp to a readable format
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

genre_columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
genre_popularity = data.groupby('gender')[genre_columns].sum().sum().sort_values(ascending=False)
print("Most popular movie genres:\n", genre_popularity)

genre_popularity.plot(kind='bar', figsize=(10, 6))
plt.title('Genre Popularity')
plt.xlabel('Genre')
plt.ylabel('Number of Ratings')
# plt.show()

average_ratings = data.groupby('gender')['rating'].mean()
print("Average ratings by gender:\n", average_ratings)

occupation_counts = data['occupation'].value_counts()
print("Distribution of users by occupation:\n", occupation_counts)

top_rated_movies = data.groupby('title')['rating'].mean().sort_values(ascending=False).head(10)
print("Top-rated movies:\n", top_rated_movies)

age_distribution = data['age'].value_counts().sort_index()
print("Age distribution of users:\n", age_distribution)

genre_popularity = data.groupby('gender')[genre_columns].sum()

#genre with the highest sum for males and females
most_popular_genre_male = genre_popularity.loc['M'].idxmax()
most_popular_genre_female = genre_popularity.loc['F'].idxmax()

print("Most popular genre for males:", most_popular_genre_male)
print("Most popular genre for females:", most_popular_genre_female)

drama_movies = data[data['Drama'] == 1]

#number of males and females who rated drama movies
males_rated_drama = drama_movies[drama_movies['gender'] == 'M']['gender'].count()
females_rated_drama = drama_movies[drama_movies['gender'] == 'F']['gender'].count()

print("Number of males who rated drama movies:", males_rated_drama)
print("Number of females who rated drama movies:", females_rated_drama)

a = open("u.user","r")

lines = []

for l in a:
  lines.append(l.split("|"))

# print(lines)

df = pd.DataFrame(lines,columns=['userId','age','gender','occupation','zipCode'])

df['age'] = df['age'].astype('int')
df.describe()
df.shape

print(df.describe())

df1 = df[df['age'] >= 20 ]

df2 = df1[df1['age']<=60]


merged_data = pd.merge(ratings, movies, on='movie_id')

users_merged = pd.merge(merged_data, users, on='user_id')

print(merged_data.head())
print(users_merged.head())

# two csv convert to one csv
users_merged.to_csv("final_data.csv")

data_df = pd.read_csv('final_data.csv')

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Generate the TF-IDF matrix using movie titles
tfidf_matrix = vectorizer.fit_transform(data_df['title'])


# Function to get movie recommendations for a user
def get_user_recommendations_content_by_age(filtered_users, user_id, tfidf_matrix, data_df):
    
    # Get the movies rated by the user
    user_movies = data_df[data_df['user_id'] == user_id]['movie_id']
    
    # Create a set of movies already rated by the user
    rated_movies = set(user_movies)
    
    # Initialize an empty set to store recommended movies
    recommended_movies = set()

    if user_id in filtered_users['user_id'].values:
        for movie_id in user_movies:
            idx = data_df[(data_df['movie_id'] == movie_id) & (data_df['user_id'].isin(filtered_users['user_id']))].index[0]

            # Calculate the pairwise cosine similarity between the current movie and all other movies
            similarity_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

            # Get the indices of movies sorted by similarity scores in descending order
            sorted_indices = similarity_scores.argsort()[::-1]

            # Get the top 5 similar movies (excluding the input movie itself)
            top_similar_movies = sorted_indices[1:6]

            # Get the movie_ids of the top similar movies
            similar_movies = data_df['movie_id'].iloc[top_similar_movies].values

            # Add the similar movies to the recommended list (excluding those already rated by the user)
            recommended_movies.update([movie for movie in similar_movies if movie not in rated_movies])
        
        return list(recommended_movies)[:5]  # Return the top 5 recommended movie_ids
    else:
        return None

filtered_users = users_merged[(users_merged['age'] >= 18) & (users_merged['age'] <= 40)]
input_user_id = 160
if input_user_id in filtered_users['user_id'].values:
    recommendations = get_user_recommendations_content_by_age(filtered_users, input_user_id, tfidf_matrix, data_df)
    print("Recommended movies for user with ID '{}':".format(input_user_id))
    for movie_id in recommendations:
        movie_title = data_df[data_df['movie_id'] == movie_id]['title'].iloc[0]
        print(movie_title)
else:
    print("Input user ID '{}' not found in the specified age range.".format(input_user_id))



# Create a Surprise Dataset from the DataFrame
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_df[['user_id', 'movie_id', 'rating']], reader)

# Split the data into train and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Define and train the user-based collaborative filtering model
model = KNNBasic(sim_options={'user_based': True})
model.fit(trainset)

# Function to get movie recommendations for a user
def get_user_recommendations_collabrative(filtered_users, user_id, model, data_df):
    # Get all unique movie IDs
    if user_id in filtered_users['user_id'].values:
        all_movie_ids = data_df['movie_id'].unique()
        
        # Get the movie IDs not rated by the user
        user_rated_movies = data_df[data_df['user_id'] == user_id]['movie_id']
        user_unrated_movies = [movie_id for movie_id in all_movie_ids if movie_id not in user_rated_movies]
        
        # Predict ratings for the unrated movies
        predictions = [model.predict(user_id, movie_id) for movie_id in user_unrated_movies]
        
        # Sort the predictions by predicted rating in descending order
        sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
        
        # Get the top 5 recommended movie IDs
        recommended_movies = [pred.iid for pred in sorted_predictions[:5]]
    
        return recommended_movies
    
    else:
        return None

filtered_users = users_merged[(users_merged['age'] >= 18) & (users_merged['age'] <= 40)]
input_user_id = 417
if input_user_id in filtered_users['user_id'].values:
    recommendations = get_user_recommendations_collabrative(filtered_users, input_user_id, model, data_df)
    print("Recommended movies for user with ID '{}':".format(input_user_id))
    for movie_id in recommendations:
        movie_title = data_df[data_df['movie_id'] == movie_id]['title'].iloc[0]
        print(movie_title)
else:
    print("Input user ID '{}' not found in the specified age range.".format(input_user_id))


