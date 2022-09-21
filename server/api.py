import flask
from flask import Flask, request
import json
import pandas as pd
from inference import predict

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.get("/")
def home():
    return 'Greetings my dear hackathon team, the server is up and running !'


@app.get("/score")
def get_score():
    data = request.args
    form = pd.DataFrame(data, index=[0])
    # prediction = predict(form)
    message = "Received data and processing it later"
    response = flask.jsonify({'data': message})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)