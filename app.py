import os
from wsgiref import simple_server

from flask import Flask, request, render_template, jsonify

from bidirectional_lstm_rnn_sentiment_analysis_rotten_tomatoes import RottenTomatoesLSTM

app = Flask(__name__)


class ClientApp:
    def __init__(self):
        self.tokenizer = 'models/sentiments_100neurons_300dim_60000voc_100seq_tokenizer'
        self.model = 'models/sentiments_100neurons_300dim_60000voc_100seq_model-17_0.54'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    sentence = request.form['sentence']
    params = {'embedding_dim': 300,
             'vocabulary_size': 60000,
             'seq_size': 100,
             'nb_epochs': 50,
             'batch_size': 128,
             'memory_neurons': 100,
             'target': 'Sentiment',
             'samples': 62500}

    lstm = RottenTomatoesLSTM(params)
    lstm.load(clApp.tokenizer, clApp.model)
    predicted_sentiment = lstm.make_prediction(sentence)

    return render_template('index.html',
                           prediction_text='Sentiment of the provided review is : {}'.format(predicted_sentiment))


@app.route('/predict-api', methods=['POST'])
def predictApi():
    '''
    For rendering results on HTML GUI
    '''
    sentence = request.form['sentence']
    params = {'embedding_dim': 300,
             'vocabulary_size': 60000,
             'seq_size': 100,
             'nb_epochs': 50,
             'batch_size': 128,
             'memory_neurons': 100,
             'target': 'Sentiment',
             'samples': 62500}

    lstm = RottenTomatoesLSTM(params)
    lstm.load(clApp.tokenizer, clApp.model)
    predicted_sentiment = lstm.make_prediction(sentence)

    return jsonify({"sentiment":predicted_sentiment})


port = int(os.getenv("PORT", 8080))
if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=port)
    clApp = ClientApp()
    #host = "127.0.0.1"
    host = '0.0.0.0'
    #port = 5000
    # app.run(host='0.0.0.0', port=port, app=app)
    httpd = simple_server.make_server(host, port, app)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
    app.run(debug=True)
