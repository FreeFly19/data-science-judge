import json
import os
import time
from os.path import exists
from threading import Lock

from flask import Flask, render_template, request, jsonify, send_file
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import numpy as np

app = Flask(__name__)

os.makedirs('data', exist_ok=True)
submissions_json = 'data/submissions.json'
if not exists(submissions_json):
    with open(submissions_json, 'w') as f:
        json.dump([], f)

with open(submissions_json, 'r') as f:
    submissions = json.load(f)

X, y = fetch_openml("optdigits", version=1, return_X_y=True, as_frame=False)
json_data = X.tolist()
y = y.astype(int)
del X

@app.get('/')
def index():
    public_board_user_best = {}
    for s in submissions:
        if s['user_name'] not in public_board_user_best:
            public_board_user_best[s['user_name']] = s['public_board_score']
        if public_board_user_best[s['user_name']] < s['public_board_score']:
            public_board_user_best[s['user_name']] = s['public_board_score']

    public_user_scores = []
    for user_name in public_board_user_best:
        public_user_scores.append({'user_name': user_name, 'score': public_board_user_best[user_name]})

    public_user_scores = list(sorted(public_user_scores, key=lambda us: -us['score']))


    private_board_user_best_submission= {}

    for s in submissions:
        if s['user_name'] not in private_board_user_best_submission:
            private_board_user_best_submission[s['user_name']] = s
        if private_board_user_best_submission[s['user_name']]['private_board_score'] < s['private_board_score']:
            private_board_user_best_submission[s['user_name']] = s

    private_user_best_submissions = []
    for user_name in private_board_user_best_submission:
        private_user_best_submissions.append({'user_name': user_name, 'submission': private_board_user_best_submission[user_name]})

    private_user_best_submissions = list(sorted(private_user_best_submissions, key=lambda us: -us['submission']['private_board_score']))

    return render_template('index.html', public_user_scores=public_user_scores, private_user_best_submissions=private_user_best_submissions)


@app.get('/data')
def get_date():
    return jsonify(json_data)

@app.get('/solutions/<int:id>/code')
def get_solution(id):
    return send_file('data/' + str(id) + '/code.py')

lock = Lock()
@app.post('/submit')
def predict():
    with lock:
        user_name = request.json['user_name']
        predictions = request.json['predictions']
        code = request.json['code']

        public_board_score = accuracy_score(y[:len(y)//2], np.array(predictions)[:len(y)//2])
        private_board_score = accuracy_score(y[len(y)//2:], np.array(predictions)[len(y)//2:])

        id = len(submissions)
        submissions.append({
            'id': id,
            'time': int(time.time()),
            'user_name': user_name,
            'public_board_score': public_board_score,
            'private_board_score': private_board_score,
        })
        os.makedirs('data/' + str(id), exist_ok=True)
        with open('data/' + str(id) + '/predictions.json', 'w') as f_preds:
            json.dump(predictions, f_preds)
        with open('data/' + str(id) + '/code.py', 'w') as f_code:
            f_code.write(code)
        with open(submissions_json, 'w') as f:
            json.dump(submissions, f)
        return jsonify({'score': public_board_score})


app.run(host='0.0.0.0', port=8989, threaded=True)
