from flask import Flask, request
import json


app = Flask(__name__)


@app.get("/")
def home():
    return 'Server up and running'


@app.get("/score")
def get_score():
    args = request.args
    

if __name__ == '__main__':
    app.run(debug=True)