import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import laplace_asymmetric
from scipy.optimize import minimize
import sys

# Simulation of prices and the premium for a reserve price,
# given a dataframe with base fee and date, and a strike price we will simulate the future prices and calculate the premium
# nPeriods is the number of hours in the future to simulate, num_paths is the number of paths to simulate
# cap_level is the cap level in basis points, risk_free_rate is the risk-free rate
# ===============================================================================================
def simulate_prices_and_payoff(df, strike, num_paths=15000, nPeriods=720, cap_level=0.3, risk_free_rate=0.05):

    # Data Cleaning and Preprocessing - removing NaN if exist and log transformation
    # ===============================================================================================
    df.dropna(inplace=True)
    df['log_base_fee'] = np.log(df['base_fee'])
    df.reset_index(inplace=True)

    # Running a linear regression to discover the trend, then removing that trend from the log base fee
    # ===============================================================================================
    df['time_index'] = np.arange(len(df))
    X = np.column_stack([df['time_index'], np.ones(len(df))])
    y = df['log_base_fee']
    trend_model = sm.OLS(y, X).fit()
    df['trend'] = trend_model.predict(X)
    df['detrended_log_base_fee'] = df['log_base_fee'] - df['trend']

    # plt.plot(df['date'], df['detrended_log_base_fee'], label='Log Base Fee', color='black')
    # plt.show()

    # Seasonality modelling and removal from the detrended log base fee
    # ===============================================================================================
    start_date = df['date'].iloc[0]
    df['t'] = (df['date'] - start_date).dt.total_seconds() / 3600
    def season_matrix(t):
        return np.column_stack((                        
            np.sin(2 * np.pi * t / 24),  
            np.cos(2 * np.pi * t / 24),
            np.sin(4 * np.pi * t / 24), 
            np.cos(4 * np.pi * t / 24),
            np.sin(8 * np.pi * t / 24), 
            np.cos(8 * np.pi * t / 24),
            np.sin(2 * np.pi * t / (24 * 7)),  
            np.cos(2 * np.pi * t / (24 * 7)),
            np.sin(4 * np.pi * t / (24 * 7)),  
            np.cos(4 * np.pi * t / (24 * 7)),
            np.sin(8 * np.pi * t / (24 * 7)),  
            np.cos(8 * np.pi * t / (24 * 7)),
        ))
    C = season_matrix(df['t'])
    seasonParam, _, _, _ = np.linalg.lstsq(C, df['detrended_log_base_fee'], rcond=None)
    season = C @ seasonParam
    df['de_seasonalized_detrended_log_base_fee'] = df['detrended_log_base_fee'] - season


    # Monte Carlo Parameter Estimation for the MRJ model
    # ===============================================================================================
    dt = 1 / (365 * 24)
    Pt = df['de_seasonalized_detrended_log_base_fee'].values[1:]
    Pt_1 = df['de_seasonalized_detrended_log_base_fee'].values[:-1]
 

    def laplace_pdf(x, mu, b):
        return 1 / (2 * b) * np.exp(-np.abs(x - mu) / b)

    def mrjpdf(params, Pt, Pt_1, dt):
        a, phi, mu_J, sigmaSq, sigmaSq_J, lambda_ = params
        term1 = lambda_ * laplace_pdf(Pt, a * dt + phi * Pt_1 + mu_J, np.sqrt(sigmaSq + sigmaSq_J))
        term2 = (1 - lambda_) * laplace_pdf(Pt, a * dt + phi * Pt_1, np.sqrt(sigmaSq))

        return term1 + term2
    
    def neg_log_likelihood(params, Pt, Pt_1, dt):
        pdf_vals = mrjpdf(params, Pt, Pt_1, dt)
        # Avoid log(0) by adding a small constant to pdf_vals
        log_likelihood = np.sum(np.log(pdf_vals + 1e-1))
        return -log_likelihood
    
    x0 = [0, 0, 0.7, np.var(df['de_seasonalized_detrended_log_base_fee']), 0.1, 0.005]
    # x0 = [-0.10687699614209123, 1623.0894971158705, -7.862335551687283, -3494.3353532126216, -1.6678530639386735, -1268.804171506291]
    
    print("\n")
    print("X0", x0)
    print("Dt", dt)
    print("Pt", Pt)
    print("Pt_1", Pt_1)
    print("\n")

    initial_neg_log_likelihood = neg_log_likelihood(x0, Pt, Pt_1, dt)
    print("Initial negative log-likelihood:", initial_neg_log_likelihood)

    bounds = [
        (-np.inf, np.inf), 
        (-np.inf, 1), 
        (0, np.inf), 
        (0, np.inf), 
        (0, np.inf), 
        (0, 24)
        ]
    result = minimize(neg_log_likelihood, x0, args=(Pt, Pt_1, dt), bounds=bounds, method='L-BFGS-B')
    params = result.x
    alpha = params[0] / dt
    kappa = (1 - params[1]) / dt
    mu_J = params[2]
    sigma = np.sqrt(params[3] / dt)
    sigma_J = np.sqrt(params[4]) 
    lambda_ = params[5] / dt

    print("Fitted params", params)

    return "",""

    # Monte Carlo Simulation of the MRJ model
    # ===============================================================================================
    nPeriods = 30 * 24 
    j = np.random.binomial(1, lambda_ * dt, (nPeriods, num_paths))
    SimPrices = np.zeros((nPeriods, num_paths))
    SimPrices[0, :] = df['de_seasonalized_detrended_log_base_fee'].values[-1]
    n1 = np.random.normal(size=(nPeriods, num_paths)) #fitted from previous ARIMA analysis.
    n2 = np.random.normal(size=(nPeriods, num_paths))
    for i in range(1, nPeriods):
        SimPrices[i, :] = alpha * dt + (1 - kappa * dt) * SimPrices[i - 1, :] + sigma * np.sqrt(dt) * n1[i, :] + j[i, :] * (mu_J + sigma_J * n2[i, :])
    last_date = df['date'].iloc[-1]
    SimPriceDates = pd.date_range(last_date, periods=nPeriods, freq='H')
    SimHourlyTimes = (SimPriceDates - df['date'].iloc[0]).total_seconds() / 3600

    # Adding seasonality back to the simulated prices
    # ===============================================================================================
    C = season_matrix(SimHourlyTimes)
    season = C @ seasonParam
    logSimPrices = SimPrices + season.reshape(-1, 1)


    # Calibrating and adding stochastic trend to the simulation. 
    # =============================================================================================== 
    df['log_TWAP_7d'] = np.log(df['TWAP_7d'])
    returns = df['log_TWAP_7d'].diff().dropna()
    mu = 0.05 / 52 # Weekly drift
    sigma = returns.std() * np.sqrt(24 * 7)  # Weekly volatility
    dt = 1 / 24  # Time step in hours
    stochastic_trend = np.zeros((nPeriods, num_paths))
    for i in range(num_paths):
        random_shocks = np.random.normal(0, sigma * np.sqrt(dt), nPeriods)
        stochastic_trend[:, i] = np.cumsum((mu - 0.5 * sigma**2) * dt + random_shocks)
    
    # Adding trend and stochastic trend to the simulation, considering the final trend value
    # =============================================================================================== 
    SimLogPricesWithTrend = np.zeros_like(logSimPrices)
    coeffs = trend_model.params
    final_trend_value = np.polyval(coeffs, len(df) - 1)
    for i in range(nPeriods):
        trend = final_trend_value  # Use the final trend value for all future time points
        SimLogPricesWithTrend[i, :] = logSimPrices[i, :] + trend + stochastic_trend[i, :]
    simulated_log_df = pd.DataFrame(SimLogPricesWithTrend, index=SimPriceDates, columns=[f'Trial_{i+1}' for i in range(num_paths)])
    simulated_df = np.exp(simulated_log_df)
    simulated_df_twap = simulated_df.rolling(window=24 * 7).mean()
    simulated_df_twap.dropna(inplace=True)
    final_prices_twap = simulated_df_twap.iloc[-1]

    # Option Pricing and Present Value Calculation
    # ===============================================================================================    
    payoffs = np.maximum(np.minimum((1 + cap_level) * strike, final_prices_twap) - strike, 0)
    average_payoff = np.mean(payoffs)
    present_value = np.exp(-risk_free_rate) * average_payoff
    
    # Ignore! Just for plotting
    # ===============================================================================================    
    # Plot the prices and simulated prices
    PricesSim = np.exp(SimLogPricesWithTrend)
    df['base_fee'] = np.exp(df['log_base_fee'])
    plt.figure(figsize=(12, 8))
    plt.plot(df['date'], df['base_fee'], label='Actual Base Fee', color='darkgrey')
    plt.plot(df['date'], df['TWAP_7d'], label='TWAP Base Fee', color='black', linestyle='dashed')
    plt.plot(SimPriceDates, PricesSim[:, 0], label='Simulated Base Fee', color='black')
    plt.plot(simulated_df_twap.index, simulated_df_twap.iloc[:, 0], label='Simulated TWAP', color='blue', linestyle='dashed')
    plt.title('Actual Base Fee, Simulated Base Fee, and Simulated TWAP')
    plt.xlabel('Date')
    plt.ylabel('Base Fee')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    #plt.show()

    # Plot a histogram of the final simulated prices
    plt.figure(figsize=(10, 6))
    plt.hist(final_prices_twap, bins=75, edgecolor='black')
    plt.axvline(strike, color='blue', linestyle='dashed', linewidth=2, label='Mean')
    plt.title('Histogram of Final Simulated Base Fees')
    plt.xlabel('Base Fee')
    plt.ylabel('Frequency')
    plt.grid(False)
    #plt.show()

    return simulated_df_twap, present_value

# Example usage:
df = pd.read_csv('../data.csv')  # Ensure your data.csv has 'timestamp' and 'base_fee' columns
df['date'] = pd.to_datetime(df['timestamp'], unit='s')
df = df.set_index('date').resample('h').mean().reset_index()
df['TWAP_7d'] = df['base_fee'].rolling(window=24 * 7).mean()

# plt.plot(df['date'], df['TWAP_7d'])
# plt.show()

# Create a list to store the dataframes
dfs = []

start_date = df['date'].min()
end_date = df['date'].max()
num_months = ((end_date.year - start_date.year) * 12 + end_date.month - start_date.month) + 1

# Split the dataframe into overlapping 3-month periods
for i in range(num_months - 4):  # Ensure we have at least 6 months for the last period
    period_start = start_date + pd.DateOffset(months=i)
    period_end = period_start + pd.DateOffset(months=5)
    period_df = df[(df['date'] >= period_start) & (df['date'] < period_end)]
    dfs.append(period_df)

# Apply the simulation and print results
to_export = []
for idx, period_df in enumerate(dfs):
    strike = period_df['TWAP_7d'].iloc[-1]  # Strike is at the money
    simulated_prices, premium = simulate_prices_and_payoff(period_df, strike)

    break

    if idx + 1 < len(dfs):
        next_period_df = dfs[idx + 1]
        settlement_price = next_period_df['TWAP_7d'].iloc[-1]
        ending_timestamp = next_period_df['timestamp'].iloc[-1]
    else:
        pass # Or handle the last period accordingly

    to_append = {
        'starting_timestamp': round(period_df['timestamp'].iloc[-1]),
        'ending_timestamp': round(ending_timestamp),
        'reserve_price': premium,
        'strike_price': strike, 
        'settlement_price': settlement_price,  # First value of the next period
        'cap_level': 3000,  # in basis points
    }
    print(to_append)
    to_export.append(to_append)
    pd.DataFrame(to_export).to_csv('output.csv', index=False)
    print(f"Period {idx + 1}")
    print(f"Simulated Prices:\n{simulated_prices}")
    print(f"Average Payoff: {premium:.2f}\n")

print(to_export)