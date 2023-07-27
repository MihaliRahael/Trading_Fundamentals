# 3 Easy Ways to Smoothen Time Series Data

![image](https://github.com/MihaliRahael/Trading_Fundamentals/assets/106816732/65beb2d6-3a1f-4a82-9595-75d962cadbe9)

Financial data can be super noisy, and smoothing it out can be a lifesaver, helping traders and investors to:
-	Spot trends and direction (both short and long term)
-	Create strategies (like the good old moving average crossover)
-	Get it ready for machine learning models
Giving a smoother representation of your data to an ML model is like giving it a cheat sheet to learn faster

## Method 1: Moving Averages
Moving average approach may cause your data to have a delayed or lagged appearance. Why? In essence, a “conventional” moving average solely incorporates past data, hence its categorization as a “lagging indicator.” Although it is possible to use a centered window and determine the average of the midpoint.

An additional challenge arises from the fact that each data point is given equal weighting in the average calculation. This can result in an inflated average if historical prices were much higher. To resolve this issue, the exponential window grants greater weight to the most recent points in time.

But if the spread of your data is large, or worse, bimodally distributed (two peaks in your data distribution), then the mean may not be a suitable measure since the distribution of the data is not normal!
An exponential weighting is merely one method of determining how to weight the data points. In reality, there are several other techniques at your disposal, all of which can be implemented using the .rolling function with the help of the win_type keyword argument.

To illustrate, suppose you opt for a Gaussian window, which would prioritize the data points situated in the middle. Alternatively, you can select any of the windows available in the scipy.signal.windows library, which offers a wide range of options. a Gaussian weighting, and increasing the standard deviation. An increasing standard deviation means we give more weight to the points toward the ends of the window.

### My thoughts
Using moving average for smoothening the data for a trading strategy is not advisable. If you are using it, then keep in mind to 
-	Stick to a small window size – this will help reduce any lagging effects.
-	Experiment with different window types to see what works best for your data.

## Method 2: Savitzky-Golay Filter
Rather than simply averaging a group of data points, the Savitzky-Golay filter employs local polynomial interpolation to smooth out your time-series data.
Here’s the rough process in a nutshell:
-	Take a handful of data points (your chosen window size).
-	Find a smooth curve through the data points using mathematical magic. This curve is expressed as a function.
-	Evaluate that smooth curve function at the middle of the window.
Boom, that’s your smoothened data point for that window. This process is repeated for each window.
![image](https://github.com/MihaliRahael/Trading_Fundamentals/assets/106816732/f1bd1402-b750-4d70-a95a-4c7a5621e5d6)

There are two parameters to this approach:
-	The window size
It is the amount of data you’re fitting the polynomial to — **the larger the window, the smoother the data. Make sure to use an odd number for your window size**
-	The polynomial degree
The best bet is to use a lower-degree polynomial — one of degree 5 should be just fine for most purposes

## Method 3: Partial Differential Equations
Let’s dive into one more approach to smoothen time-series data – solving Partial Differential Equations (PDEs). There are two approaches.
-	The Heat Equation
-	The Perona-Malik PDE

What are the advantages of using a PDE?
-	With this approach, **we have full control over the end-points**. In the example above, I kept them fixed so that they don’t move. With other smoothening methods, you may not have this level of control and would need to make artificial adjustments after smoothening.
-	This method is **faster to run** when you use something like Numba. This makes it more computationally efficient, especially if you’re processing a lot of data.
-	**When it comes to the heat equation, there is only one parameter to adjust** – how smooth you want your time-series to be
-	**With the Perona-Malik PDE, we have an additional parameter for edge preservation**. This helps to maintain the overall shape of the data.

### IMPORTANT NOTE : 
I smoothened the entire time series first and then broke it down into smaller segments for a prediction task. I got amazing results, but the problem was that centered smoothening approaches like the Savitzky-Golay filter and PDE methods use data from both past and future points to smoothen the data. This means **there’s a risk of data leakage from future points**, which can seriously mess up your predictions.
To give an example, let’s say you have a time series of 100 points and you smoothen it. You then cut it in half to the first 50 points and ask a model to predict the 51st point – you might get a good result because the 51st point was considered when smoothening the 50th! It’s a total nightmare.
But there’s a simple solution: **break up your time-series data into smaller segments first, and then smoothen each segment individually**. This way, you won’t have any data leakage from future points.


# How to Preserve Edges when Smoothening Time-Series Data

One method we can apply to better understand trends (or to aid pattern recognition/predictive algorithms) is that of time-series smoothening. Traditional approaches are
-	Moving averages — simple, easy, and effective (but gives a “lagged” appearance to your time-series data).
-	The Savitzky-Golay filter — effective, but more complex and has somewhat intuitive hyperparameters
Here we will use Perona-Malik Partial Differential Equation (PDE) for edge preservation. The issue with the heat-equation is that it does not preserve edges very well. It may be desirable to keep these edges to capture large and quick fluctuations in price, but remove any small but high-frequency noise. This approach is harder than the heat equation because the Perona-Malik PDE is non-linear (unlike the heat equation which is linear). Generally speaking, non-linear equations are not as easy to solve as linear ones.
