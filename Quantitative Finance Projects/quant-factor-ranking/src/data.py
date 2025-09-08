import pandas as pd
import os

def load_prices(ticker, data_path="../data/raw"):
    """Load CSV and fix columns to standard OHLCV format."""
    import pandas as pd, os

    df = pd.read_csv(
        os.path.join(data_path, f"{ticker}.csv"),
        skiprows=2
    )

    # First column is Date
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # Force datetime conversion here
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), errors="coerce")
    # If any value in the "Date" column can't be converted to a valid date, it will be replaced with a NaT (Not a Time) value instead of causing an error.
    df = df.set_index("Date")

    # Rename OHLCV columns
    df.columns = ["Close", "High", "Low", "Open", "Volume"]

    # Drop any NaNs
    df = df.dropna(how="any")

    return df.sort_index()



def load_close_prices(tickers, data_path="../data/raw"):
    dfs = []
    for t in tickers:
        df = load_prices(t, data_path)
        dfs.append(df["Close"].rename(t))
    prices = pd.concat(dfs, axis=1)

    return prices
