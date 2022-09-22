import flask
from flask import Flask, request
from yscore import Yscore
import pandas as pd
from inference import predict

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
yscore = Yscore('../model/xgbr_model.pkl')


@app.get("/")
def home():
    return 'Greetings my dear hackathon team, the server is up and running !'


@app.get("/score")
def get_score():
    data = request.args
    form = pd.DataFrame(data, index=[0])
    print(yscore.improve_score(form))

    score, weights = yscore.feature_weigths(form)
    response_data = {'fico': str(score)}
    response = flask.jsonify(response_data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)