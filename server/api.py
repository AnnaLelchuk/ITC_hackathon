from flask import Flask, request
import json


app = Flask(__name__)


@app.get("/")
def home():
    return 'Greetings my dear hackathon team, the server is up and running !'


@app.get("/score")
def get_score():
    return 'Not implemented yet'
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)