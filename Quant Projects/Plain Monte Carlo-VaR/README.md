Project Title: Financial Risk Analysis: Value at Risk (VaR) and Expected Shortfall (ES)

Introduction

This project explores two fundamental concepts in financial risk management: Value at Risk (VaR) and Expected Shortfall (ES). It demonstrates how to calculate these metrics using two distinct approaches: a Monte Carlo simulation and a parametric (analytical) method. The project uses simulated data to provide a clear, reproducible example of how these risk measures quantify potential portfolio losses.



Problem Statement

The objective is to quantify the potential risk of a financial portfolio. VaR provides a single number representing the maximum expected loss at a given confidence level, while ES goes a step further by calculating the average loss in the worst-case scenarios. The project aims to accurately calculate both metrics and compare the results from the two different methodologies.



Methodology

Data Simulation: The project begins by defining and simulating a time series of asset returns based on a geometric Brownian motion model. Key parameters like the daily mean log-return and volatility are defined, and then scaled for the desired time horizon.



Monte Carlo Simulation: A large number of possible future returns are simulated using a random number generator. This creates an empirical distribution of potential losses for the portfolio. VaR is then calculated as the specific percentile of this loss distribution, and ES is computed as the average of all losses exceeding that VaR threshold.



Parametric Method: This approach assumes that portfolio returns follow a specific statistical distribution (in this case, the normal distribution). VaR and ES are calculated using analytical formulas, which are derived from the properties of the normal distribution.



Comparison and Visualization: The project compares the results from the Monte Carlo simulation and the parametric method. It also includes a histogram of the simulated loss distribution, with vertical lines marking the calculated VaR levels. This visualization provides an intuitive understanding of what VaR represents on the loss distribution.



Key Findings

VaR vs. ES: The project clearly demonstrates the difference between VaR (a point on the distribution) and ES (the average of the tail). While VaR provides a loss boundary, ES gives a more conservative measure of "tail risk."



Method Comparison: Under the assumption of normality, the results from the Monte Carlo and parametric methods are shown to be very close. This confirms the validity of both approaches for this specific problem.



Reproducibility: The use of a random seed ensures that the Monte Carlo simulation results are reproducible, which is a critical aspect of quantitative analysis.



Technologies Used

Python



Numpy



Scipy



Matplotlib

