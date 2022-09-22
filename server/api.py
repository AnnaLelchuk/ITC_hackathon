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

    score, important_params = yscore.feature_weigths(form)
    current_params_dict, new_params_dict, new_score, score_diff = yscore.improve_score(form)

    response_data = {'fico': int(score),
                     'current_params': current_params_dict,
                     'new_params': new_params_dict,
                     'new_fico': int(new_score),
                     'fico_diff': int(score_diff),
                     'most_affecting_params': important_params
                    }
    response = flask.jsonify(response_data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)