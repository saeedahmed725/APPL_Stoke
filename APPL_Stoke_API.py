import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI
import uvicorn
import yfinance as yf
import tensorflow as tf

# # Initialize FastAPI app
app = FastAPI()

# Load ML model and scaler
model = tf.keras.models.load_model("model (1).keras")
scaler = MinMaxScaler()

def fetch_last_60_minutes():
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="1d", interval="1m")
    last_60_minutes = data.tail(60)
    return last_60_minutes

# Function to scrape the real-time AAPL price
def scrape_aapl_data():
    url = "https://www.google.com/search?q=AAPL+stock+price"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract stock price
    price = soup.find("div", {"class": "BNeawe iBp4i AP7Wnd"}).text.replace(",", "")
    return price

@app.get("/predict")
def predict_stock():

    # Fetch last 60 Close prices
    last_60_closes = fetch_last_60_minutes()
    
    # Reshape and scale the data
    close_df = pd.DataFrame(last_60_closes, columns=['Open','High', 'Low', 'Close', "Volume"])
    scaled_data = scaler.fit_transform(close_df)
    X_input = scaled_data.reshape(1, 60, 5) 

    # Predict the next Close price
    predicted_price = model.predict(X_input)
    predicted_price = model.predict(X_input)

    input_for_inverse_transform = np.array([[predicted_price[0][0], 0, 0, 0, 0]])

    predicted_price_inverse = scaler.inverse_transform(input_for_inverse_transform)

    return {"predicted_price": predicted_price_inverse[0][0]}
    
    
    
  
@app.get("/current-price")
def get_current_price():
    price = scrape_aapl_data()
    return {"price": price}
  
    
    
    
@app.get("/last-60-minutes")
def get_last_60_minutes():
    data = fetch_last_60_minutes()
    data.reset_index(inplace=True)  # Reset index to access datetime
    result = [
        {
            "datetime": row["Datetime"].isoformat(),
            "open": row["Open"],
            "high": row["High"],
            "low": row["Low"],
            "close": row["Close"],
            "volume": int(row["Volume"]),
            "dividends": row["Dividends"],
            "stock_splits": row["Stock Splits"]
        }
        for _, row in data.iterrows()
    ]
    return result 
    
    

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
