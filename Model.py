# TODO Make the plot more readable
# implement TensorFlow
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # So far, only predicts thorugh changes of price. 

# Function to download and preprocess data
def preprocess_data(ticker, start_date=None, end_date=None):
    data = yf.download(ticker,start=start_date, end=end_date)
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data = data.dropna()
    
    # Feature selection
    X = data[['Close', 'MA_10', 'MA_50', 'RSI']]
    y = data['Close'].shift(-1).dropna()
    X = X[:-1]  # Remove last row to align with shifted y
    return X, y, data

# Function to calculate RSI(Reltative Strength Index)
def calculate_rsi(series, window=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to train the model
def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Function to evaluate the model
def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')
    return preds

# Function to simulate trading strategy
def trading_strategy(X_test, preds, init_balance=1000):
    balance = init_balance
    position = 0
    transaction_fee = 0.01  # 1% transaction fee(depning on market)

    for i in range(len(X_test)):
        curr_price = float(X_test['Close'].values[i])
        pred_price = float(preds[i])

        if pred_price > curr_price and balance >= curr_price:
            shares_to_buy = int(balance // curr_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * curr_price * (1 + transaction_fee)
                position += shares_to_buy
                balance -= cost
                print(f"Buying {shares_to_buy} shares at ${curr_price:.2f}, Cost: ${cost:.2f}")

        elif pred_price < curr_price and position > 0:
            proceeds = position * curr_price * (1 - transaction_fee)
            balance += proceeds
            print(f"Selling {position} shares at ${curr_price:.2f}, Proceeds: ${proceeds:.2f}")
            position = 0

    # Final balance calculation
    final_balance = balance + (position * X_test.iloc[-1]['Close'])
    profit = final_balance - init_balance
    print(f"Final balance: ${final_balance:.2f}")
    print(f"Profit: ${profit:.2f}")
    return final_balance, profit

# Main script
ticker = 'AAPL' #Stock
start_date = input("Enter the start date (YYYY-MM-DD) or press Enter to skip: ")
end_date = input("Enter the end date (YYYY-MM-DD) or press Enter to skip: ")

X, y, data = preprocess_data(ticker, start_date if start_date else None, end_date if end_date else None)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model, scaler = train_model(X_train, y_train)
preds = evaluate_model(model, scaler, X_test, y_test)

# Plot actual vs predicted prices(make it look noicer)
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test.values, label='Actual Price', alpha=0.7)
plt.plot(y_test.index, preds, label='Predicted Price', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()
plt.show()

# Execute trading strategy
trading_strategy(X_test, preds)
