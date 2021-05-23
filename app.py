import os
from flask import Flask, jsonify, request, render_template
from flask_restful import Api, Resource
from model.Train import trainModel
import joblib
import pandas as pd

app = Flask(__name__)
api = Api(app)

if not os.path.isfile('iris-model.model'):
    trainModel()

model = joblib.load('iris-model.model')


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        #file = request.form['upload-file']
        file = request.files['upload-file']
        #file.save(os.getcwd()+'/uploaded.xlsx')
        data = pd.read_excel(file)
        return render_template('index.html', data=data.to_dict())


class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json(force=True)
        sepal_length = posted_data['sepal_length']
        sepal_width = posted_data['sepal_width']
        petal_length = posted_data['petal_length']
        petal_width = posted_data['petal_width']

        prediction = model.predict(
            [[sepal_length, sepal_width, petal_length, petal_width]])[0]
        if prediction == 0:
            predicted_class = 'Iris-setosa'
        elif prediction == 1:
            predicted_class = 'Iris-versicolor'
        else:
            predicted_class = 'Iris-virginica'

        return jsonify({
            'Prediction': predicted_class
        })


api.add_resource(MakePrediction, '/predict')
if __name__ == '__main__':
    app.run(debug=True, port=8000)
