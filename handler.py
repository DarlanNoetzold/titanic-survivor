import os
from flask import Flask, request
import pandas as pd
import pickle

from titanic_data.titanic_data import TitanicData

model = pickle.load(open("modelo/modelo-titanic.pkl", 'rb'))

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    test_json = request.get_json()

    if test_json:
        if isinstance(test_json, dict):
            df_raw = pd.DataFrame(test_json, index=[0])
        else:
            df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

    pipeline = TitanicData()
    df1 = pipeline.data_preparation(df_raw)

    pred = model.predict(df1)

    df1['prediction'] = pred

    return df1.to_json(orient='records')

if __name__ == '__main__':
    port = os.environ.get('PORT', 8880)
    app.run(host='192.168.1.105', port=port)