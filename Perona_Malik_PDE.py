import numpy as np
import pandas as pd
import yfinance as yf  from typing import Tuple
from copy import deepcopy  import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default='browser'  TICKER = 'TSLA'
EMA_LENGTH = 10
N_DAYS = 100
ALPHA = 10
T_END = 4  def get_diff_mat(n: int) -> Tuple[np.array, np.array]:
 '''
 Get the first and second differentiation matrices, which are truncated
 to find the derivative on the interior points.
 Parameters
 ----------
 n : int
 The total number of discrete points
 Returns
 -------
 [Dx, Dxx] : np.array
 The first and second differentiation matrices, respecitvely.
 ''' 
 Dx = (
 np.diag(np.ones(n-1), 1) 
 - np.diag(np.ones(n-1), -1)
 )/2 
 Dxx = (
 np.diag(np.ones(n-1), 1)
 - 2*np.diag(np.ones(n), 0)
 + np.diag(np.ones(n-1), -1)
 ) 
 # Truncate the matrices so that we only determine the derivative on the
 # interior points (i.e. we don't calculate the derivative on the boundary)
 return Dx[1:-1, :], Dxx[1:-1, :]  def perona_malik_smooth(p: np.array,
 alpha: float = 10.0,
 k: float = 0.05,
 t_end: float = 5.0) -> np.array:
 '''
 Solve the Gaussian convolved Perona-Malik PDE using a basic finite 
 difference scheme.
 Parameters
 ----------
 p : np.array
 The price array to smoothen.
 alpha : float, optional
 A parameter to control how much the PDE resembles the heat equation,
 the perona malik PDE -> heat equation as alpha -> infinity
 k : float, optional
 The step size in time (keep < 0.1 for accuracy)
 t_end : float, optional
 When to termininate the algorithm the larger the t_end, the smoother
 the series
 Returns
 -------
 U : np.array
 The Perona-Malik smoothened time series
 ''' 
 Dx, Dxx = get_diff_mat(p.shape[0]) 
 U = deepcopy(p)
 t = 0  while t < t_end: 
 # Find the convolution of U with the guassian, this ensures that the
 # PDE problem is well posed
 C = convolve_PDE(deepcopy(U)) 
 # Determine the derivatives by using matrix multiplication
 Cx = Dx@C
 Cxx = Dxx@C 
 Ux = Dx@U
 Uxx = Dxx@U 
 # Find the spatial component of the PDE
 PDE_space = (
 alpha*Uxx/(alpha + Cx**2)
 - 2*alpha*Ux*Cx*Cxx/(alpha + Cx**2)**2
 ) 
 # Solve the PDE for the next time-step
 U = np.hstack((
 np.array([p[0]]),
 U[1:-1] + k*PDE_space,
 np.array([p[-1]]),
 )) 
 t += k 
 return U  def convolve_PDE(U: np.array,
 sigma: float = 1,
 k: float = 0.05) -> np.array:
 '''
 Perform Gaussian convolution by solving the heat equation with Neumann
 boundary conditions
 Parameters
 ----------
 U : np.array
 The array to perform convolution on.
 sigma : float, optional
 The standard deviation of the guassian convolution.
 k : float, optional
 The step-size for the finite difference scheme (keep < 0.1 for accuracy)
 Returns
 -------
 U : np.array
 The convolved function
 ''' 
 t = 0
 t_end = sigma**2/2 
 while t < t_end: 
 # Implementing the nuemann boundary conditions
 U[0] = 2*k*U[1] + (1-2*k)*U[0]
 U[-1] = 2*k*U[-2] + (1-2*k)*U[-1] 
 # Scheme on the interior nodes
 U[1:-1] = k*(U[2:] + U[:-2]) + (1-2*k)*U[1:-1] 
 t += k 
 return U  def heat_smooth(p: np.array,
 k: float = 0.05,
 t_end: float = 5.0) -> np.array:
 '''
 Solve the heat equation using a basic finite difference approach.
 Parameters
 ----------
 p : np.array
 The price array to smoothen.
 k : float, optional
 The step size in time (keep < 0.1 for accuracy)
 t_end : float, optional
 When to termininate the algorithm the larger the t_end, the smoother
 the series
 Returns
 -------
 U : np.array
 The heat equation smoothened time series
 ''' 
 # Obtain the differentiation matrices
 _, Dxx = get_diff_mat(p.shape[0]) 
 U = deepcopy(p)
 t = 0 
 while t < t_end:  U = np.hstack((
 np.array([p[0]]),
 U[1:-1] + k*Dxx@U,
 np.array([p[-1]]),
 )) 
 t += k 
 return U  def get_candlestick_chart(df: pd.DataFrame):
 '''
 Plot the candlestick chart with the smoothened time series + the EMA
 '''  layout = go.Layout(
 title = f'{TICKER} Stock Price',
 xaxis = {'title': 'Date'},
 yaxis = {'title': 'Price'},
 legend = {'orientation': 'h', 'x': 0, 'y': 1.075},
 width = 700,
 height = 700,
 ) 
 fig = go.Figure(
 layout=layout,
 data=[
 go.Candlestick(
 x = df['Date'],
 open = df['Open'], 
 high = df['High'],
 low = df['Low'],
 close = df['Close'],
 showlegend=False,
 ),
 go.Scatter(
 x = df['Date'], 
 y = df['ema'], 
 name = 'Exponenatial Moving Average', 
 line={'color': 'rgba(180, 207, 0, 1)'},
 mode = 'lines',
 ),
 go.Scatter(
 x = df['Date'], 
 y = df['perona_malik'], 
 name = 'Perona-Malik',
 line={'color': 'rgba(48, 0, 223, 1)'},
 mode = 'lines',
 ),
 go.Scatter(
 x = df['Date'], 
 y = df['heat'], 
 name = 'Heat Equation', 
 line={'color': 'rgba(21, 222, 215, 1)'},
 mode = 'lines',
 ),
 ]
 ) 
 fig.update_xaxes(
 rangebreaks = [{'bounds': ['sat', 'mon']}],
 rangeslider_visible = False,
 ) 
 return fig  if __name__ == '__main__': 
 df = yf.download(TICKER).reset_index() 
 df['ema'] = df['Close'].ewm(EMA_LENGTH).mean()
 df = df[-N_DAYS:] 
 df['perona_malik'] = perona_malik_smooth(
 p = df['Close'].values, 
 alpha = ALPHA, 
 k = 0.05, 
 t_end = T_END,
 )
 df['heat'] = heat_smooth(
 p = df['Close'].values, 
 k = 0.05, 
 t_end = T_END,
 ) 
 fig = get_candlestick_chart(df)
 fig.show()
