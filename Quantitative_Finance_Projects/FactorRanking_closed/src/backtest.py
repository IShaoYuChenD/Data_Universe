import pandas as pd
import numpy as np

def backtest_topk(pred_ranks: pd.DataFrame, returns: pd.DataFrame, k: int = 2) -> pd.Series:
    """
    Backtest a top-k portfolio strategy using predicted ranks and realized returns.

    Parameters
    ----------
    pred_ranks : DataFrame
        Predicted probabilities (higher = more likely to be top performer).
        Index = dates (monthly), Columns = tickers.
    returns : DataFrame
        Realized returns (aligned with pred_ranks).
        Same index and columns as pred_ranks.
    k : int
        Number of top assets to hold each period.

    Returns
    -------
    portfolio_returns : Series
        Portfolio returns over time.
    """

    # --- 1. Select top-k assets each period ---
    topk_mask = pred_ranks.rank(axis=1, ascending=False) <= k # perform the ranking row by row
    weights = topk_mask.astype(int).div(topk_mask.sum(axis=1), axis=0) 
    #  True becomes 1, and False becomes 0. Divides each 1 in the weights DataFrame by the sum of its row (which is k)

    # --- 2. Compute portfolio returns ---
    portfolio_returns = (returns * weights).sum(axis=1)

    return portfolio_returns
