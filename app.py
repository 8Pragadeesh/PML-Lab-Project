import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# Load pickled model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a Flask app
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    return render_template('home.html')

# Define route for prediction page
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction == 0:
        prediction='DrugA'
    elif prediction == 1:
        prediction='DrugB'
    elif prediction == 2:
        prediction='DrugC'
    elif prediction == 3:
        prediction='DrugX'
    elif prediction == 4:
        prediction='DrugY'
    
    return render_template('predict.html', prediction=prediction)

# Run the app
if __name__ == '__main__':
    app.run(host='localhost',debug=True,port=8000)
