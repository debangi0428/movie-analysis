from flask import Flask, render_template, request
from datetime import datetime
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import sqlite3

# Create the Flask application
app = Flask(__name__)

#This empty till will store my data from the request form below
user_data = []

# Define a route and its corresponding function

data_df = pd.read_csv('final_data.csv')

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Generate the TF-IDF matrix using movie titles
tfidf_matrix = vectorizer.fit_transform(data_df['title'])

# Create a Surprise Dataset from the DataFrame
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_df[['user_id', 'movie_id', 'rating']], reader)

# Split the data into train and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Define and train the user-based collaborative filtering model
# model = KNNBasic(sim_options={'user_based': True})
# model.fit(trainset)

# Train the model with the updated data
trainset = data.build_full_trainset()
model = KNNBasic(sim_options={'user_based': True})
model.fit(trainset)

def get_user_recommendations_collabrative(age, gender, occupation, movie_id, rating, data_df):

    #create new user id after all the existing user
    new_user_id = data_df['user_id'].max() + 1

    print({'user_id': [new_user_id],
            'age': [age] })

    new_user_ratings = pd.DataFrame({'user_id': [new_user_id],
                                     'age': [age],
                                     'gender': [gender],
                                     'occupation': [occupation],
                                     'movie_id': [movie_id],
                                     'rating': [rating] })
    
        
    updated_data_df = pd.concat([data_df, new_user_ratings], ignore_index=True)

    # Create a Surprise Dataset from the updated DataFrame
    updated_data = Dataset.load_from_df(updated_data_df[['user_id', 'movie_id', 'rating']], reader)

    trainset_new = updated_data.build_full_trainset()

    # Use the existing model for predictions
    trainset_new = updated_data.build_full_trainset()
    model_new = KNNBasic(sim_options={'user_based': True})
    model_new.fit(trainset_new)
    # model_new.train(trainset_new)

    predictions = [model_new.predict(new_user_id, movie_id).est for movie_id in updated_data_df['movie_id']]

    # Sort the predictions by predicted rating in descending order
    sorted_predictions = sorted(zip(updated_data_df['movie_id'], predictions), key=lambda x: x[1], reverse=True)

    # Get the top 5 recommended movie IDs
    # recommended_movies = ','.join([str(movie_id) for movie_id, _ in sorted_predictions[:5]])
    recommended_movies = []
    for movie_id, _ in sorted_predictions:
        if movie_id != 50 and movie_id not in recommended_movies:
            recommended_movies.append(movie_id)
            if len(recommended_movies) == 5:
                break

    recommended_movies_info = []
    for movie_id in recommended_movies:
        movie_info = data_df[data_df['movie_id'] == movie_id][['movie_id', 'title', 'IMDb_URL']].iloc[0]
        recommended_movies_info.append((movie_info['movie_id'], movie_info['title'], movie_info['IMDb_URL']))

    return recommended_movies_info
    
    # return recommended_movies

# render recs to template, call this after 
# @app.route('/response')
# def response(recommendations):
#     # return "<h1>"+recommendations+"</h1>"
#     table_html = "<table>"
#     table_html += "<tr><th>Movie ID</th><th>Title</th><th>IMDb URL</th></tr>"
#     for movie_id, title, IMDb_URL in recommendations:
#         table_html += f"<tr><td>{movie_id}</td><td>{title}</td><td><a href='{IMDb_URL}'>{IMDb_URL}</a></td></tr>"
#     table_html += "</table>"
#     return table_html

@app.route('/response')
def response(recommendations):
    return render_template("response.html", recommendations=recommendations)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve input data from the request form
        name = request.form.get('name')
        age = int(request.form.get('age'))
        email = request.form.get('email')
        gender = request.form.get('gender')
        occupation = request.form.get('occupation')
        rating = int(request.form.get('rating'))

        # Validate email address
        email_pattern = r'^[a-zA-Z0-9_.+-]+@gmail\.com$'
        if not re.match(email_pattern, email):
            error_message = "Invalid email. Email must end in '@gmail.com'."
            return render_template("./form.html", error=error_message)

        if int(age) < 18:
            error_message = "You must be at least 18 years old."
            return render_template("./form.html", error=error_message)

        # recommendations = get_user_recommendations_collabrative(age, gender, occupation, rating, data_df)
        movie_id = int(request.form.get('movie_id'))
        recommendations = get_user_recommendations_collabrative(age, gender, occupation, movie_id, rating, data_df)
        if recommendations:
            return response(recommendations)

    # If it's a GET request, render an HTML form for input
    return render_template("./form.html")

# Run the Flask application
if __name__ == '__main__':
    app.run(port=5000)
