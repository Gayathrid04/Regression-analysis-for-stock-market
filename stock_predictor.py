import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import plotly.graph_objs as go

class StockPredictor:
    def __init__(self):
        self.window_size = 60
        self.model = self._build_model()
        self.scaler = MinMaxScaler(feature_range=(-1,1))
    
    def _build_model(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.window_size, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def get_stock_data(self, ticker):
        df = yf.download(ticker, start="2012-01-01", end=datetime.now())
        return df
    
    def predict(self, ticker):
        # Get data
        df = self.get_stock_data(ticker)
        data = df[['Close']].values
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        
        # Prepare training data
        training_data_len = int(np.ceil(len(data) * .95))
        train_data = scaled_data[0:training_data_len, :]
        
        # Create sequences
        x_train, y_train = [], []
        for i in range(self.window_size, len(train_data)):
            x_train.append(train_data[i-self.window_size:i, 0])
            y_train.append(train_data[i, 0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Train model
        self.model.fit(x_train, y_train, epochs=25, batch_size=32, validation_split=0.1, verbose=0)
        
        # Prepare test data
        test_data = scaled_data[training_data_len - self.window_size:, :]
        x_test = []
        y_test = data[training_data_len:]
        
        for i in range(self.window_size, len(test_data)):
            x_test.append(test_data[i-self.window_size:i, 0])
            
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        # Make predictions
        predictions = self.model.predict(x_test)
        predictions = self.scaler.inverse_transform(predictions)

        self.print_accuracy_metrics(predictions, y_test)
        
        return predictions, y_test
    
    def print_accuracy_metrics(self, predictions, actual_values):
        rmse = np.sqrt(np.mean((predictions - actual_values) ** 2))
        mape = np.mean(np.abs((predictions - actual_values) / actual_values)) * 100
        mae = np.mean(np.abs(predictions - actual_values))

        print(f"RMSE: {rmse}")
        print(f"MAPE: {mape}%")
        print(f"MAE: {mae}")
    
    def create_price_chart(self, df, predictions, actual_values):
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(go.Scatter(
            x=df.index[-len(actual_values):],
            y=actual_values.flatten(),
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Add predicted prices
        fig.add_trace(go.Scatter(
            x=df.index[-len(predictions):],
            y=predictions.flatten(),
            mode='lines',
            name='Predicted',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Stock Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price USD ($)',
            template='plotly_white'
        )
        
        return fig.to_html(full_html=False)