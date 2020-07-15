"""
Description: Flask app for Restaurant review sentimental analysis

@author: Kishorlal
"""
import os
import pickle

# Importing Flask library
from flask import Flask, render_template, request

# Create flask object
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__),r"templates"))

# Saved path of our model
MODEL_PATH = os.path.join(os.path.dirname(__file__),r"sentimentalAnalysis.pkl")

# Count vectorizer path
cvFilePath=os.path.join(os.path.dirname(__file__),r"countvectorizer.pkl")

# Load our classifier and count vectorizer
classifier = pickle.load(open(MODEL_PATH, 'rb'))
cv = pickle.load(open(cvFilePath, 'rb'))
 
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vector = cv.transform(data).toarray()
    	modelPrediction = classifier.predict(vector)
    	return render_template('result.html', prediction=modelPrediction)

if __name__ == '__main__':
	app.run(debug=True)