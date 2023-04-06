import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# Load pickled model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model_reg.pkl', 'rb') as f1:
    model_reg = pickle.load(f1)

# Create a Flask app
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/classify')
def classify():
    return render_template('classify.html');

# Define route for prediction page
@app.route('/validate', methods=['GET','POST'])
def validate():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    if prediction == 0:
        prediction='Your Tablet Name is "DrugA"'
    elif prediction == 1:
        prediction='Your Tablet Name is "DrugB"'
    elif prediction == 2:
        prediction='Your Tablet Name is "DrugC"'
    elif prediction == 3:
        prediction='Your Tablet Name is "DrugX"'
    elif prediction == 4:
        prediction='Your Tablet Name is "DrugY"'
    
    return render_template('classify.html', prediction=prediction)

@app.route('/reg')
def reg():
    return render_template('reg.html');
    
@app.route('/validates', methods=["GET","POST"])
def validates():
    int_features =[int (x) for x in request.form.values()]
    final_features =[(np.array (int_features))]
    print(final_features)
    prediction=model_reg.predict(final_features)
    output=np.round(prediction[0],2)
    
    return render_template('reg.html',prediction='Fish Weight is {} in grams'.format(output))
   

# Run the app
if __name__ == '__main__':
    from gunicorn.app.wsgiapp import run
    run()
