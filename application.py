from flask import Flask, render_template, request
from datetime import datetime
import re

# Create the Flask application
app = Flask(__name__)

#This empty till will store my data from the request form below
user_data = []

# Define a route and its corresponding function
@app.route('/response')
def response(name, age):
    current_time = datetime.now().time()
    return render_template('./index.html', user=name, time=current_time, age=age)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve input data from the request form
        name = request.form.get('name')
        age = request.form.get('age')
        email = request.form.get('email')
        gender = request.form.get('gender')
        city = request.form.get('city')
        country = request.form.get('country')

        # Validate email address
        email_pattern = r'^[a-zA-Z0-9_.+-]+@gmail\.com$'
        if not re.match(email_pattern, email):
            error_message = "Invalid email. Email must end in '@gmail.com'."
            return render_template("./form.html", error=error_message)

        if int(age) < 18:
            error_message = "You must be at least 18 years old."
            return render_template("./form.html", error=error_message)


        # Store the form data in the list
        user_data.append({
            'name': name,
            'age': age,
            'email': email,
            'gender': gender,
            'city': city,
            'country': country
        })

        # Process the input data
        result = "Hello, " + name + "! Welcome to the home page."
        return result
        # Return the response
        # return response(name)

    # If it's a GET request, render an HTML form for input
    return render_template("./form.html")

# Run the Flask application
if __name__ == '__main__':
    app.run(port=5000)
