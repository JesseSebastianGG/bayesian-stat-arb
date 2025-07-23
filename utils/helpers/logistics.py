import pandas as pd


def get_ticker_pairs(tickers):
    n = len(tickers)
    perms = []
    ticker_pairs = []
    for i in range(n-1):
        for j in range(i+1, n):
            perms.append([i, j])
            ticker_1 = tickers[i]
            ticker_2 = tickers[j]
            ticker_pairs.append([ticker_1, ticker_2])
    return ticker_pairs


def describe_to_markdown(classical, bayesian, round_decimals=3):
    """
    Takes two pandas Series or DataFrames of .describe() output
    and prints a Markdown-formatted comparison table.
    """
    combined = pd.concat([classical, bayesian], axis=1)
    combined.columns = ['Classical', 'Bayesian']
    combined = combined.round(round_decimals)

    md = "| Metric | Classical | Bayesian |\n"
    md += "|--------|-----------|----------|\n"
    for index, row in combined.iterrows():
        md += f"| {index.capitalize()} | {row['Classical']} | {row['Bayesian']} |\n"
    print(md)
