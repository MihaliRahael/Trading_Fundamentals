import yfinance as yf  # Download the daily stock-price history for Tesla
df = yf.download('TSLA')  # Calculate the moving averages and store as a new columns
df['MA'] = df['Close'].rolling(20).mean()
df['MA_gauss'] = df['Close'].rolling(20, win_type='gaussian').mean(std=2)  # Filter to the last 120 trading days and plot
df = df[-120:]
df.plot(y=['Close', 'MA', 'MA_gauss'])
