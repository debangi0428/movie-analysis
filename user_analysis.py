import pandas as pd

# a = open("u.user","r")

# lines = []

# for l in a:
#   lines.append(l.split("|"))

# # print(lines)

# df = pd.DataFrame(lines,columns=['userId','age','gender','occupation','zipCode'])

# df['age'] = df['age'].astype('int')
# df.describe()
# df.shape

# df1 = df[df['age'] >= 20 ]

# df2 = df1[df1['age']<=60]

# df2.describe()

# df2.head()

# b = open("u.data","r")
# b_lines = []
# for l in b:
#   b_lines.append(l.split("\t"))

# rating_df = pd.DataFrame(b_lines,columns=['userId','titleId','rating','timeStamp'])
# rating_df_1 = rating_df[rating_df['userId'].isin(df2['userId']) ]

# c = open("u.item","r",encoding="ISO-8859-1")

# cols = ['id','title','releaseDate','videoReleaseDate','imdbURL', "unknown" ,"Action" , "Adventure" , "Animation" , "Childrens" , "Comedy" , "Crime" , "Documentary" , "Drama" , "Fantasy" ,
#         "Film-Noir" , "Horror" , "Musical" , "Mystery" , "Romance" , "Sci-Fi" ,
#         "Thriller" , "War" , "Western"]

# c_lines = []

# for l in c:
#   c_lines.append(l.strip().split("|"))

# df = pd.DataFrame(c_lines,columns=cols)

# df[(df['Adventure']=='1') & (df['Action']=='1')]



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
#IMDn_URL is missing for 13
# release date is missing for 9

# Drop unnecessary columns
data = data.drop(['video_release_date', 'IMDb_URL', 'unknown'], axis=1)

# Convert timestamp to a readable format
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

genre_columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
genre_popularity = data.groupby('gender')[genre_columns].sum().sum().sort_values(ascending=False)
print("Most popular movie genres:\n", genre_popularity)

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



