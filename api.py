from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
from chat_functions import predict_class
from dotenv import load_dotenv
import os

load_dotenv()

USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
PORT = os.getenv("PORT")

app = Flask(__name__)
auth = HTTPBasicAuth()


@auth.verify_password
def verify_password(username, password):
    if username == USERNAME and password == PASSWORD:
        return username


@app.route("/intents")
@app.errorhandler(400)
@auth.login_required
def get_intents():
    sentence = request.args.get("sentence")

    if sentence is None:
        return {
            "status_code": 400,
            "error": "Bad Request",
            "message": "Please provide a non-nullable 'sentence' query param"
        }

    intents = predict_class(sentence)
    return jsonify(intents)


if __name__ == "__main__":
    app.run(port=PORT, threaded=False, debug=False)
