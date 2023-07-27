import yfinance as yf  # Download the daily stock-price history for Tesla
df = yf.download('TSLA')  # Calculate the moving average and store as a new column
df['MA'] = df['Close'].rolling(20).mean()  # Filter to the last 120 trading days and plot
df = df[-120:]
df.plot(y=['Close', 'MA'])

# to calculate a exponential moving average, just swap out the .rolling for a .ewm.
