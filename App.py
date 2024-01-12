from flask import Flask, render_template
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)


# Scaricare i dati storici delle aziodddni
@app.route('/')
def index():
    # Scaricare i dati storici delle aziodddni
    ticker = 'AAPL'  # Esempio: Apple Inc.
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01")

    # Preparare i dati per la previsione
    data['Prediction'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    X = np.array(data[['Close']])
    Y = np.array(data['Prediction'])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Creare e addestrare il modello di regressione lineare
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Testare il modello
    confidence = model.score(X_test, Y_test)

    # Prevedere il prezzo delle azioni
    X_future = data['Close'].tail(1)
    X_future = np.array(X_future).reshape(-1, 1)
    predicted_price = model.predict(X_future)[0]

    return render_template('index.html', predizione=predicted_price, fiducia=confidence)

if __name__ == '__main__':
    app.run(debug=True)
