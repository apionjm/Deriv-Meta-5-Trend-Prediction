import MetaTrader5 as mt5
import keras
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import LeakyReLU
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
from tensorflow.keras.models import load_model
from datetime import datetime
from tensorflow.keras.models import save_model


# Initialize MetaTrader5
mt5.initialize()

# Global parameters
symbol = "Volatility 10 Index"
timeframe = mt5.TIMEFRAME_M1
num_candles = 10000

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='live-candlestick', config={'displayModeBar': True}, style={'width': '100%'}),
        dcc.Graph(id='trend-chart', config={'displayModeBar': True}, style={'width': '100%'})
    ], style={'display': 'flex', 'width': '100%'}),
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)  # Update every 5 seconds
])

# Function to fetch MT5 data
def fetch_mt5_data(symbol, timeframe, num_candles):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    return data

# Function to add technical indicators
def add_technical_indicators(data):
    data['ma_10'] = data['close'].rolling(window=10).mean()
    data['ma_50'] = data['close'].rolling(window=50).mean()
    data['rsi'] = 100 - (100 / (1 + (data['close'].diff().apply(lambda x: max(x, 0)).rolling(window=14).mean() /
                                     data['close'].diff().apply(lambda x: abs(min(x, 0))).rolling(window=14).mean())))
    fast_ema = data['close'].ewm(span=12, adjust=False).mean()
    slow_ema = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = fast_ema - slow_ema
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['atr'] = (data['high'] - data['low']).rolling(window=14).mean()
    return data.dropna()

# Save Model with Versioning
def save_model_with_version(model, history_dict, iteration, backup_dir="models_backup"):
    # Ensure the backup directory exists
    os.makedirs(backup_dir, exist_ok=True)

    # Get timestamp for versioning
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Define model and history file paths with iteration and timestamp
    model_filename = os.path.join(backup_dir, f"model_iter_{iteration}_{timestamp}.h5")
    history_filename = os.path.join(backup_dir, f"history_iter_{iteration}_{timestamp}.json")

    # Save the model in HDF5 format
    model.save(model_filename)

    # Save training history as a JSON file
    with open(history_filename, 'w') as f:
        json.dump(history_dict, f)

    print(f"Backup saved: Model as {model_filename} and History as {history_filename}")

# Create Model
def create_model(input_shape):
    model = models.Sequential([
        # First hidden layer with regularization
        layers.Dense(256, activation='relu', input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.4),
        LeakyReLU(alpha=0.3),
        layers.BatchNormalization(),

        # Second hidden layer with regularization
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.4),
        layers.BatchNormalization(),

        # Third hidden layer with regularization
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.4),
        layers.BatchNormalization(),

        # Fourth hidden layer
        layers.Dense(32, activation='relu'),

        # Fifth hidden layer
        layers.Dense(16, activation='relu'),

        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model with an optimizer and loss function
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Train the machine learning model
def train_model():
    data = fetch_mt5_data(symbol, timeframe, 300)
    data = add_technical_indicators(data)
    model_path = "aitradePredict.h5"

    X = data[['close', 'ma_10', 'ma_50', 'rsi', 'macd', 'macd_signal', 'atr']].values
    y = (data['close'].shift(-10) > data['close']).astype(int).values  # Predict next price movement

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
# Load existing model or create a new one
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Loaded existing model from disk.")
        
        # Compile the model again after loading
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        print("No existing model found. Creating a new one.")
        model = create_model((X_train_scaled.shape[1],))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}


# EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # Create the model and train
    input_shape = (X_train_scaled.shape[1],)
    model = create_model(input_shape)


    # Train the model
    history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, validation_data=(X_test_scaled, y_test))

    # Save the model and history after training
    save_model_with_version(model, history.history, iteration=1)  # You can replace 1 with the actual iteration count or loop number
# Save the trained model
    save_model(model, model_path)
    #model.save(model_path)
    print("Model saved successfully.")

    return model, scaler

# Train the model once and reuse
model, scaler = train_model()

# Function to create candlestick chart with trend direction arrow
def create_candlestick_chart(data):
    # Get the trend prediction (up or down) based on the last data point
    X = data[['close', 'ma_10', 'ma_50', 'rsi', 'macd', 'macd_signal', 'atr']].values
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled[-1:])[0][0]
    trend_strength = (y_pred - 0.5) * 2  # Positive for uptrend, negative for downtrend

    # Determine the trend direction
    trend_direction = "up" if trend_strength >= 0.5 else "down"

    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data['time'], open=data['open'], high=data['high'], low=data['low'], close=data['close']
    )])

    # Add an arrow indicating trend direction
    arrow_color = "green" if trend_direction == "up" else "red"
    arrow_marker = dict(symbol='triangle-up' if trend_direction == "up" else 'triangle-down', 
                        color=arrow_color, size=10)

    # Add an arrow at the latest candle's close timing
    fig.add_trace(go.Scatter(
        x=[data['time'].iloc[-1]],
        y=[data['close'].iloc[-1]],
        mode='markers',
        marker=arrow_marker,
        name=f'Trend: {trend_direction.capitalize()}'
    ))

    fig.update_layout(title="Live Candlestick Chart", xaxis_title="Time", yaxis_title="Price", autosize=True)
    return fig


# Function to create trend prediction chart with success markers
def create_trend_chart(data):
    X = data[['close', 'ma_10', 'ma_50', 'rsi', 'macd', 'macd_signal', 'atr']].values
    X_scaled = scaler.transform(X)

    future_steps = 2
    predicted_prices = []
    last_close = data['close'].iloc[-1]
    actual_trend = data['close'].shift(-1) > data['close']  # Actual trend direction

    success_markers = []
    failed_markers = []

    for i in range(future_steps):
        y_pred = model.predict(X_scaled[-1:])[0][0]
        trend_strength = (y_pred - 0.5) * 2
        next_price = last_close * (1 + (trend_strength * 0.001))
        predicted_prices.append(next_price)
        last_close = next_price

        # Compare with actual trend
        if i < len(actual_trend) - 1:  # Ensure valid index
            actual_move = actual_trend.iloc[-(future_steps - i)]
            predicted_move = y_pred > 0.5  # Uptrend if > 0.5
            success = actual_move == predicted_move

            marker_color = "green" if success else "red"
            marker_list = success_markers if success else failed_markers
            marker_list.append(
                go.Scatter(
                    x=[data['time'].iloc[-1] + pd.Timedelta(minutes=5 * (i + 1))],
                    y=[predicted_prices[-1]],
                    mode='markers',
                    marker=dict(symbol='circle', color=marker_color, size=10),
                    name='Success' if success else 'Failed Prediction'
                )
            )

    future_times = [data['time'].iloc[-1] + pd.Timedelta(minutes=5 * (i + 1)) for i in range(future_steps)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['time'], y=data['close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=future_times, y=predicted_prices, mode='lines+markers', line=dict(dash='dash', color='blue'), name='Predicted Trend'))
    #train_model()
    #save_model(model, model_path)
    # Add success and failed markers
    for marker in success_markers + failed_markers:
        fig.add_trace(marker)

    fig.update_layout(title="Trend Prediction with Success Markers", xaxis_title="Time", yaxis_title="Price", autosize=True)
    return fig

@app.callback(
    [Output('live-candlestick', 'figure'),
     Output('trend-chart', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_charts(n):
    # Fetch latest data
    data = fetch_mt5_data(symbol, timeframe, num_candles)
    data = add_technical_indicators(data)

    # Generate charts
    candlestick_chart = create_candlestick_chart(data)
    trend_chart = create_trend_chart(data)

    return candlestick_chart, trend_chart

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
