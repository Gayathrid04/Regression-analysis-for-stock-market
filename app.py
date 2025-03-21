from flask import Flask, render_template, request
from stock_predictor import StockPredictor
import pandas as pd
import numpy as np
app = Flask(__name__)
predictor = StockPredictor()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        ticker = request.form['ticker']
        try:
            # Get stock data and predictions
            df = predictor.get_stock_data(ticker)
            predictions, actual_values = predictor.predict(ticker)
            
            # Get the chart
            chart = predictor.create_price_chart(df, predictions, actual_values)
            
            # Calculate metrics
            rmse = float(np.sqrt(np.mean((predictions - actual_values) ** 2)))
            mape = float(np.mean(np.abs((actual_values - predictions) / actual_values)) * 100)
            mae = float(np.mean(np.abs(predictions - actual_values)))
            
            return render_template('index.html', 
                                chart=chart, 
                                ticker=ticker,
                                prediction_made=True,
                                rmse=f"${rmse:.2f}",
                                mape=f"{mape:.2f}%",
                                mae=f"${mae:.2f}")
        except Exception as e:
            error_message = f"Error processing ticker {ticker}: {str(e)}"
            return render_template('index.html', error=error_message)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)