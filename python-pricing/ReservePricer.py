import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ruptures import Binseg
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import laplace_asymmetric


# Simulation of prices and the premium for a reserve price,
# given a dataframe with base fee and date, and a strike price we will simulate the future prices and calculate the premium
# nPeriods is the number of hours in the future to simulate, num_paths is the number of paths to simulate
# cap_level is the cap level in basis points, risk_free_rate is the risk-free rate
# ===============================================================================================
def simulate_prices_and_payoff(df, strike, num_paths=15000, nPeriods=720, cap_level=0.3, risk_free_rate=0.05):

    # print(df.head(1))
    # print(df.tail(1))
    # print(df[np.isnan(df["TWAP_7d"])])
    # rows_with_nan = df[df.isna().any(axis=1)]
    # print(rows_with_nan)
    # Data Cleaning and Preprocessing - removing NaN if exist and log transformation
    # ===============================================================================================
    df.dropna(inplace=True)
    df['log_base_fee'] = np.log(df['base_fee'])
    df.reset_index(inplace=True )

    # Running a linear regression to discover the trend, then removing that trend from the log base fee
    # ===============================================================================================
    df['time_index'] = np.arange(len(df))
    X = np.column_stack([df['time_index'], np.ones(len(df))])
    y = df['log_base_fee']
    trend_model = sm.OLS(y, X).fit()
    df['trend'] = trend_model.predict(X)
    df['detrended_log_base_fee'] = df['log_base_fee'] - df['trend']

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
    

    # Fitting an ARIMA model to the deseasaonalized and detrended log base fee, finding a distribution for the residuals
    # ===============================================================================================
    DTMdl = ARIMA(df['de_seasonalized_detrended_log_base_fee'], order=(12, 0, 4))
    results = DTMdl.fit()
    df['fitted'] = results.fittedvalues
    df['residules'] = df['de_seasonalized_detrended_log_base_fee'] - results.fittedvalues
    condVar = np.var(df['residules'])
    df['standardized_residuals'] = df['residules'] / np.sqrt(condVar)
    kappa, loc, scale = laplace_asymmetric.fit(df['standardized_residuals'])

    # The code below is just for debugging - to be removed

    # new_df = df[['de_seasonalized_detrended_log_base_fee']].head(10)
    # DTMdl = ARIMA(new_df['de_seasonalized_detrended_log_base_fee'], order=(12, 0, 4))
    # results = DTMdl.fit()
    # print(results.summary())
    # new_df['fitted'] = results.fittedvalues
    # new_df['residules'] = new_df['de_seasonalized_detrended_log_base_fee'] - results.fittedvalues
    # print(new_df)

    print(results.summary())
    return True, True

    # Simulating future prices using the ARIMA model and the distribution of the residuals 
    # ===============================================================================================
    start_date = df['date'].iloc[-1]
    sim_standardized_residuals = laplace_asymmetric.rvs(kappa=kappa, loc=loc, scale=scale, size=(nPeriods, num_paths))
    sim_residuals = sim_standardized_residuals * np.sqrt(condVar)
    arima_forecast = results.get_forecast(steps=nPeriods)
    arima_predictions = arima_forecast.predicted_mean.values  
    simulated_paths = np.zeros((nPeriods, num_paths))
    for i in range(num_paths):
        simulated_paths[:, i] = arima_predictions + sim_residuals[:, i]
    last_date = df['date'].iloc[-1]
    SimPriceDates = pd.date_range(last_date, periods=nPeriods, freq='H')
    SimHourlyTimes = (SimPriceDates - df['date'].iloc[0]).total_seconds() / 3600

    # Adding seasonality back to the simulated prices
    # ===============================================================================================
    C = season_matrix(SimHourlyTimes)
    season = C @ seasonParam
    logSimPrices = simulated_paths + season.reshape(-1, 1)


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
df['TWAP_7d'] = df['base_fee'].rolling(window=24*7).mean()


# plt.plot(df['date'], df['TWAP_7d'])
# plt.show()

# # Create a list to store the dataframes
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
    # if idx + 1 < len(dfs):
    #     next_period_df = dfs[idx + 1]
    #     settlement_price = next_period_df['TWAP_7d'].iloc[-1]
    #     ending_timestamp = next_period_df['timestamp'].iloc[-1]
    # else:
    #     pass # Or handle the last period accordingly

    # to_append = {
    #     'starting_timestamp': round(period_df['timestamp'].iloc[-1]),
    #     'ending_timestamp': round(ending_timestamp),
    #     'reserve_price': premium,
    #     'strike_price': strike, 
    #     'settlement_price': settlement_price,  # First value of the next period
    #     'cap_level': 3000,  # in basis points
    # }
    # print(to_append)
    # to_export.append(to_append)
    # pd.DataFrame(to_export).to_csv('output.csv', index=False)
    # print(f"Period {idx + 1}")
    # print(f"Simulated Prices:\n{simulated_prices}")
    # print(f"Average Payoff: {premium:.2f}\n")

print(to_export)