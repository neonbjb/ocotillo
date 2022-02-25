from flask import Flask

app = Flask(__name__)

@app.route('/stt')
def stt():
    return "got request"
