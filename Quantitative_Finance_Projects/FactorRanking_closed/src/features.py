import pandas as pd

def make_features(prices, returns, volumes, tickers):
    """
    Construct factor features for each ticker at monthly frequency:
    - mom_1m, mom_3m, mom_6m
    - vol_3m
    - liq_3m
    Returns: MultiIndex DataFrame (date × [ticker, feature])
    """
    features = pd.DataFrame(index=prices.resample("M").last().index) 
    # The index of this empty DataFrame (a designed-container) is a DatetimeIndex containing only the last day of each month from "prices"'s original indices 

    for t in tickers:
        monthly = prices[t].resample("M").last() # a pandas Series that holds the end-of-month closing prices for a single stock ticker

        # Momentum: MultiIndex-top level is the ticker symbol, followed by the feature name
        # Any trading decision for March must be based on data available before or on the first day of the month.
        # Therefore, to avoid look-ahead bias in time-series analysis, shift the month down by one by shift(1)
        features[(t, "mom_1m")] = monthly.pct_change().shift(1)
        features[(t, "mom_3m")] = monthly.pct_change(3).shift(1)
        features[(t, "mom_6m")] = monthly.pct_change(6).shift(1)

        # Volatility (3-month rolling std of daily returns)
        features[(t, "vol_3m")] = (
            returns[t].rolling(63).std().resample("M").last()
        )  # standard deviation of daily returns over a 63-day rolling window (~ a quarter/3 months)

        # Liquidity (3-month rolling avg daily volume) --- proxied by trading volume
        features[(t, "liq_3m")] = (
            volumes[t].rolling(63).mean().resample("M").last()
        ) # average daily volume over the same 63-day rolling window

    return features.dropna(how="any") # the operations above will certainly produce NaN, which are removed here
