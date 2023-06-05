import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

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

ratings_matrix = users_merged.pivot_table(index='movie_id', columns='user_id', values='rating')
ratings_matrix = ratings_matrix.fillna(0)

user_similarity = cosine_similarity(ratings_matrix)
new_user_ratings = {'movie_id': [], 'rating': []}
new_user_ratings['movie_id'].extend([1001, 2003, 3005, 8472, 6790, 6500, 9100])  # Add movie IDs
new_user_ratings['rating'].extend([4, 3, 2, 3, 5, 3, 4])  # Add corresponding ratings

new_user_df = pd.DataFrame(new_user_ratings)

# Concatenate new user ratings with existing data
users_merged_with_new_user = pd.concat([users_merged, new_user_df])

new_user_ratings_matrix = users_merged_with_new_user.pivot_table(index='user_id', columns='movie_id', values='rating', fill_value=0)

# Transpose new_user_ratings_matrix to align dimensions for dot product
new_user_ratings_matrix = new_user_ratings_matrix.T

# Compute item similarity matrix using complete ratings matrix
item_similarity = cosine_similarity(ratings_matrix.T)

# Compute hybrid scores for the new user using new_user_ratings_matrix and item_similarity
hybrid_scores = new_user_ratings_matrix.dot(item_similarity)

# Sort the hybrid scores and get the indices of the top recommendations
top_indices = np.argsort(hybrid_scores.values)[-1][::-1][:10]  # Retrieve recommendations for the new user

top_movies = movies[movies['movie_id'].isin(top_indices)]['title'].unique()

print("Top Movie Recommendations:")
print(top_movies)