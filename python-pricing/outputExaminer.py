import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('output.csv')
print(df)

df['net_payoff'] = np.maximum( np.minimum( ( 1 +  0.3 ) * df['strike_price'], df['settlement_price'] ) - df['strike_price'], 0) - df['reserve_price']
df['cumulative_net_payoff'] = df['net_payoff'].cumsum()

plt.figure(figsize=(10, 6))
plt.plot(df['ending_timestamp'], df['settlement_price'], label='Net Payoff')
plt.plot(df['ending_timestamp'], df['reserve_price'], label='Strike Price')
plt.xlabel('Ending Timestamp')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(df['ending_timestamp'], df['cumulative_net_payoff'], label='Cumulative Net Payoff')
plt.axhline(0, color='red', linestyle='dashed', linewidth=2, label='Zero Line')
plt.xlabel('Ending Timestamp')
plt.ylabel('Cumulative Net Payoff')
plt.title('Cumulative Net Payoff Over Time')
plt.legend()
plt.show()
print(df)
