Project Title: Black-Scholes Option Pricing and Greeks Analysis



Introduction:

This project provides a comprehensive implementation of the Black-Scholes-Merton model for pricing European-style options. It goes beyond simple price calculation by generating constants from real-world stock data (Apple, AAPL) and visualizing the behavior of option prices in relation to key variables. It also computes the Option Greeks, which are essential risk management measures for derivatives.



Problem Statement:

The primary goal is to determine the theoretical fair value of a European call and put option at a specific point in time (t=0). A secondary goal is to analyze how the option's value changes in response to fluctuations in its underlying parameters, such as the stock price, volatility, time to maturity, and interest rates.



Methodology:



Data Acquisition and Parameter Generation:



Historical stock data for a specified ticker (AAPL) is downloaded using the yfinance library.



The current stock price (S 

0

​

&nbsp;) is set as the most recent closing price.



Annualized volatility (σ) is calculated from the historical daily logarithmic returns, a key input for the Black-Scholes model.



Black-Scholes Option Pricing:



The core of the project is the implementation of the Black-Scholes formula, which models the price of an option based on five key inputs:



Underlying stock price (S 

0

​

&nbsp;)



Strike price (K)



Time to maturity (T)



Risk-free interest rate (r)



Volatility (σ)



The project defines functions to calculate the prices of both call options and put options.



Visualization and Sensitivity Analysis:



The project uses matplotlib to visualize how option prices change as a function of each key input parameter. This includes plots showing the relationship between option price and the underlying stock price, volatility, strike price, risk-free rate, and time to maturity. These visualizations are crucial for understanding the sensitivities of option pricing.



Option Greeks Calculation:



The project computes the first-order Greeks—Delta, Gamma, Vega, Theta, and Rho—which are partial derivatives of the option price with respect to each of the key input variables. These measures are vital for understanding and managing the risks associated with an options portfolio.



Key Findings:



Call vs. Put Price: As the underlying stock price increases, the call price increases while the put price decreases.



Volatility: Higher volatility leads to a higher value for both call and put options, as it increases the probability of a large price movement in either direction.



Time Decay (Theta): The value of both call and put options decreases as time to maturity approaches zero, a phenomenon known as time decay.



Interest Rates: A higher risk-free rate generally increases the value of a call option and decreases the value of a put option.



Technologies Used:



Python



Pandas



NumPy



yfinance



scipy.stats



matplotlib

