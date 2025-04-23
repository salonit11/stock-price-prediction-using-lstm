# Lululemon Stock Price Prediction

This project aims to predict the future stock price of Lululemon using historical stock data, financial indicators, and machine learning techniques. The model architecture leverages a Long Short-Term Memory (LSTM) neural network, a type of recurrent neural network (RNN), for time series forecasting. The dataset is extracted from Yahoo Finance, and additional financial features are engineered to enhance the model's performance.

## Project Overview

- **Data Source:** Yahoo Finance
- **Model Architecture:** LSTM with Dropout layers for regularization
- **Performance Metrics:** 
  - Mean Absolute Error (MAE): 11.72
  - R2 Score: 0.80
- **Libraries Used:** 
  - `tensorflow`
  - `sklearn`
  - `talib`
  - `ta`
  - `numpy`
  - `pandas`
  - `yfinance`

## Feature Engineering

To improve model performance, the following features were added to the dataset:

- **RSI (Relative Strength Index)**
- **MACD (Moving Average Convergence Divergence)**
- **Bollinger Bands** (Upper and Lower Bands)

These features provide insights into market trends, momentum, and volatility, which help improve predictions of stock price movement.

## Data Preprocessing

- **Normalization:** MinMax scaling was applied to the dataset to normalize the features.
- **Lookback Window:** A lookback window of 50 days was used to feed the model with past 50 days' data for each prediction.

## Model Architecture

The model is composed of the following layers:

1. **LSTM Layer:** A Long Short-Term Memory (LSTM) layer for capturing temporal dependencies in the time series data.
2. **Dropout Layer:** Dropout layers are added after each LSTM layer for regularization to prevent overfitting.
3. **Dense Layer:** The final fully connected layer that outputs the predicted stock price.

The model is trained on the historical stock data and is validated on the test set to evaluate its performance.

## Final Dataset for Testing

The final dataset for testing was constructed by taking the last 100 columns of the training dataset and using them as input to get predictions from the test set. This approach helps simulate real-world forecasting where only past data is available for predictions.

## Performance

The model achieved the following evaluation metrics:

- **Mean Absolute Error (MAE):** 11.72
- **R2 Score:** 0.80

These results suggest that the model is able to make reasonably accurate predictions with a good level of explanatory power.

## Installation

To run this project locally, you will need to install the required libraries. You can do so using the following command:

```bash
pip install tensorflow scikit-learn talib ta numpy pandas yfinance
```
## Usage
Clone this repository:
git clone https://github.com/salonit11/stock-price-prediction-using-lstm.git

## Results

After training the model, the predictions will be plotted alongside the actual stock prices. The results include:
- Stock price prediction vs. actual price.
- Evaluation metrics such as MAE and R2 score.

## Conclusion

This model provides a valuable tool for forecasting the Lululemon stock price based on historical data and technical indicators. Further improvements can be made by tuning the hyperparameters, adding more features, or using other advanced deep learning models.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
