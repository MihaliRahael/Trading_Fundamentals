import yfinance as yf
from scipy.signal import savgol_filter  # Download the daily stock-price history for Tesla
df = yf.download('TSLA')  # Calculate the moving average and store as a new column
df['MA'] = df['Close'].rolling(21).mean()
df['sav_gol'] = savgol_filter(df['Close'], window_length = 21, polyorder = 5)  # Filter to the last 120 trading days and plot
df = df[-120:]
df.plot(y=['Close', 'MA', 'sav_gol'])
