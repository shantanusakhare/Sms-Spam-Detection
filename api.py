from flask import Flask , request, jsonify
import pickle
import numpy as np






app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"



tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


@app.route('/predict' ,methods=['POST'])
def predict():

    input_sms = request.form.get('sms')
    
    input_query  = np.array([[input_sms]])
    input_query = np.array([input_sms, dtype=float])
    result = model.predict(input_query)[0]



    return jsonify({'prediction':str(result)})




if __name__ == '__main__':
    app.run(debug=True)
