import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)['Close']
    returns = stockData.pct_change().dropna()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

# DATA À FOURNIR
stockList = ['AAPL', 'MSFT', 'KO', 'VWCE.DE', 'TTE']
stocks = stockList

endDate = dt.datetime.now()
startDate = dt.datetime(2025, 4, 1)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

mc_sims = 100
T = 100
initialPortfolio = 2000

meanM = np.full((T, len(weights)), meanReturns).T
portfolio_sims = np.zeros((T, mc_sims))

L = np.linalg.cholesky(covMatrix)

for m in range(mc_sims):
    Z = np.random.normal(size=(T, len(weights)))
    dailyReturns = meanM + np.dot(L, Z.T)
    portfolio_sims[:, m] = np.cumprod(
        np.dot(weights, dailyReturns) + 1
    ) * initialPortfolio

plt.plot(portfolio_sims)
plt.xlabel("Days")
plt.ylabel("Portfolio Value ($)")
plt.title("Monte Carlo Simulation of Liams Portfolio")
plt.show()



