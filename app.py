import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/enviar', methods=['POST'])
def enviar():
    file = request.files['csvFile']
    df = pd.read_csv(file)
    print(df)

    df['Semestre'] = df['Semestre'].astype(int)
    df['Notas'] = df['Notas'].astype(int)

    x = df['Semestre'].values.reshape(-1, 1)
    y = df['Notas'].values

    modelo = LinearRegression()
    modelo.fit(x, y)

    proximo_semestre = x.max() + 1
    previsao = modelo.predict(np.array([[proximo_semestre]]))[0]


    return render_template('previsao.html', proximo_semestre=proximo_semestre, previsao=previsao)

if __name__ == '__main__':
    app.run(debug=True)
