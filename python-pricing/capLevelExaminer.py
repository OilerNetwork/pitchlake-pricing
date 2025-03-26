import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Load the data from CSV file, assuming 'timestamp' and 'base_fee' columns exist
df = pd.read_csv("../data.csv")
# Convert the 'timestamp' column to datetime format and set it as the index
df["date"] = pd.to_datetime(df["timestamp"], unit="s")
df = df.set_index("date").resample("h").mean().reset_index()
# Convert 'base_fee' to gwei for readability
df["base_fee_gwei"] = df["base_fee"] / 1e9  # Convert base fee from wei to GWEI
df["base_fee"] = df["base_fee"] / 1e18  # Convert base fee from wei to ETH

# Calculate the 30-day TWAP of the base fee
df["TWAP_30d"] = (
    df["base_fee"].rolling(window=24 * 30).mean()
)  # 24 hours per day, 30 days
df["TWAP_30d_gwei"] = df["base_fee_gwei"].rolling(window=24 * 30).mean()

# Calculate 30-day returns
df["returns_30d"] = df["TWAP_30d"] / df["TWAP_30d"].shift(24 * 30) - 1
df = df.dropna(subset=["returns_30d"])

# Calculate volatility over 90 day windows
df["volatility_90d"] = df["returns_30d"].rolling(window=24 * 90).std()
df = df.dropna(subset=["volatility_90d"])

# Calculate 180-day rolling max returns
df["max_returns_180d"] = df["returns_30d"].rolling(window=24 * 180).max()
df.dropna(subset=["max_returns_180d"], inplace=True)

### CAP LEVEL VALUES ###


# Define the cap level function using max returns and k
def cap_level_max_returns(max_returns, k, alpha):
    lam = 1 * max_returns
    return np.maximum(((lam - k) / (alpha * (1 + k))), 0)


# Define the cap level function using volatility and k
def cap_level_volatility(volatility, k, alpha):
    lam = 2.33 * volatility
    return np.maximum(((lam - k) / (alpha * (1 + k))), 0)


def calculate_cap_levels(volatility, k_values, alpha_values):
    ##"""Calculate cap levels for different k and alpha values."""
    print("\nCap Levels for Different Parameters:")
    print("------------------------------------")
    print(f"Using volatility value: {volatility:.4f}")
    print("k values (strike price adjustment)")
    print("alpha values (target % of maximum likely return)")
    print("\nResults:")
    print("k\\alpha", end="")

    for alpha in alpha_values:
        print(f"\t{alpha:.2f}", end="")
    print("\n" + "-" * (8 + 8 * len(alpha_values)))

    for k in k_values:
        print(f"{k:.2f}", end="")
        for alpha in alpha_values:
            cap_level = cap_level_volatility(volatility, k, alpha)
            print(f"\t{cap_level:.2f}", end="")
        print()


# Calculate using volatility of 50%
volatility = 0.5
k_values = np.array([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
alpha_values = np.array([0.1, 0.25, 0.5, 0.75, 1.0])  # 10% to 100% of max returns

calculate_cap_levels(volatility, k_values, alpha_values)

print("\nCap Levels for Different k and α Values:")
print(f"Using volatility value: {volatility:.4f}")
print("k (BPS) | α (BPS) | Cap Level | Cap Level (%) | Cap Level (BPS)")
print("-" * 70)

for k in k_values:
    for alpha in alpha_values:
        cap_level = cap_level_volatility(volatility, k, alpha)
        bps = int(cap_level * 10000)
        formatted_bps = f"{bps:,}".replace(",", "_")
        print(
            f"{k:7.2f} | {alpha:7.2f} | {cap_level:9.2f} | {cap_level * 100:9.2f}% | {formatted_bps:>9}"
        )

### PLOTTING ###


# Set the font to Georgia for all plots
plt.rcParams["font.family"] = "Georgia"

# Create a directory for plots if it doesn't exist
os.makedirs("plots/extra", exist_ok=True)

### PLOT CAP LEVELS ###


# Apply the function to calculate cap levels
k = 0
alpha = 0.25
df["cap_level_max_returns"] = df["max_returns_180d"].apply(
    cap_level_max_returns, k=k, alpha=alpha
)
df["cap_level_volatility"] = df["volatility_90d"].apply(
    cap_level_volatility, k=k, alpha=alpha
)

# Create a figure and axis for the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the cap levels on the primary y-axis
ax1.plot(
    df["date"],
    df["cap_level_max_returns"],
    label="Max Returns Cap Level",
    color="orange",
)
ax1.plot(
    df["date"], df["cap_level_volatility"], label="Volatility Cap Level", color="red"
)
ax1.set_title("Max Returns Cap Level vs. Volatility Cap Level")
ax1.set_xlabel("Date")
ax1.set_ylabel("Cap Level")
ax1.legend(loc="upper left")
# Plot the basefee & twap on the secondary y-axis
ax2 = ax1.twinx()
ax2.plot(df["date"], df["base_fee_gwei"], label="Base Fee", color="black", alpha=0.6)
ax2.plot(df["date"], df["TWAP_30d_gwei"], label="Base Fee", color="blue", alpha=0.6)
ax2.set_ylabel("Basefee/Twap (GWEI)")
ax2.legend(loc="upper right")
plt.savefig("plots/cap_level_max_returns_vs_volatility.png")
plt.close()


### EXTRA: PLOT TWAP, RETURNS, MAX RETURNS, VOLATILITY ###


# 30-day TWAP
plt.figure(figsize=(10, 6))
plt.plot(df["date"], df["TWAP_30d"], label="30-Day TWAP", color="#333333")
plt.title("30-Day TWAP of Base Fee")
plt.xlabel("Date")
plt.ylabel("TWAP (ETH)")
plt.legend()
plt.savefig("plots/extra/twap_30d.png")
plt.close()

# 30d returns
plt.figure(figsize=(10, 6))
plt.plot(
    df["date"],
    df["returns_30d"],
    label="30-Day Rolling Max Returns",
    color="#333333",
)
plt.title("30-Day Rolling Returns")
plt.xlabel("Date")
plt.ylabel("30-Day returns")
plt.legend()
plt.savefig("plots/extra/returns_30d.png")
plt.close()

# Max 30d returns rolling 180d
plt.figure(figsize=(10, 6))
plt.plot(
    df["date"],
    df["max_returns_180d"],
    label="180-Day Rolling Max Returns",
    color="#333333",
)
plt.title("Max Returns Rolling 180 Days")
plt.xlabel("Date")
plt.ylabel("Max returns")
plt.legend()
plt.savefig("plots/extra/max_returns_180d.png")
plt.close()

# Volatility of 30d returns rolling 90d
plt.figure(figsize=(10, 6))
plt.plot(
    df["date"],
    df["volatility_90d"],
    label="90-Day Volatility",
    color="#333333",
)
plt.title("Volatility Rolling 90 Days")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.savefig("plots/extra/volatliity_90d.png")
plt.close()

### PLOT DISTRIBUTIONS ###


# Plot the distribution of 30-day log returns
plt.figure(figsize=(10, 6))
sns.histplot(df["returns_30d"], bins=50, kde=True, color="#333333")
plt.title("Distribution of 30-Day Log Returns")
plt.xlabel("30-Day Log Returns")
plt.ylabel("Frequency")
plt.savefig("plots/extra/returns_distribution.png")
plt.close()

# Plot the distribution max returns rolling 180 days using a histogram
plt.figure(figsize=(10, 6))
sns.histplot(df["max_returns_180d"], bins=50, kde=True, color="#333333")
plt.title("Distribution of 30-Day Rolling Max Returns")
plt.xlabel("30-Day Rolling Max Returns")
plt.ylabel("Frequency")
plt.savefig("plots/extra/rolling_max_returns_distribution.png")
plt.close()

# Plot the distribution of volatility of returns rolling 90 days using a histogram
# Plot the distribution max returns rolling 180 days using a histogram
plt.figure(figsize=(10, 6))
sns.histplot(df["volatility_90d"], bins=50, kde=True, color="#333333")
plt.title("Distribution of 30-Day Rolling Max Returns")
plt.xlabel("30-Day Rolling Max Returns")
plt.ylabel("Frequency")
plt.savefig("plots/extra/volatility_distribution.png")
plt.close()


# Plot: Base Fee & TWAP
fig, ax1 = plt.subplots(figsize=(12, 6))

# Titles and layout
fig.suptitle("Base Fee vs Cap Levels", fontsize=14)
fig.tight_layout()

# Base Fee on primary y-axis
ax1.plot(df["date"], df["base_fee"], label="Base Fee", color="grey", alpha=0.6)
ax1.plot(df["date"], df["TWAP_30d"], label="TWAP", color="black", alpha=0.6)
ax1.set_ylabel("GWEI", color="black")
ax1.tick_params(axis="y", labelcolor="black")

# Cap Levels on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(
    df["date"],
    df["cap_level_max_returns"],
    label="(Max Returns)",
    color="orange",
    linestyle="--",
)
ax2.plot(
    df["date"],
    df["cap_level_volatility"],
    label="(Volatility)",
    color="red",
    linestyle="--",
)
ax2.set_ylabel("Cap Level", color="black")
plt.savefig("plots/extra/base_fee_vs_cap_levels.png")
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
plt.close()

print("Calculating strike prices and options...")
# Liquidity in ETH
L = 1
# Translate to vaults that would have been triggered - assuming ATM vault here.
df["strike"] = df["TWAP_30d"].shift(24 * 30) * (
    1 + k
)  # Shift the TWAP_30d by 30 days to get the strike price (ATM)
df["previous_cap_level_volatility"] = df["cap_level_volatility"].shift(
    24 * 30
)  # Shift the cap level by 30 days
df["previous_collateral_per_option"] = (
    df["previous_cap_level_volatility"] * df["strike"]
)  # Calculate the collateral per option
df["prev_options_minted"] = (
    L / df["previous_collateral_per_option"]
)  # Calculate the number of options that can be minted

print("Calculating payoffs...")
# Calculate the payoff and total payoff as a percentage of liquidity
df["payoff"] = np.maximum(
    np.minimum((1 + df["previous_cap_level_volatility"]) * df["strike"], df["TWAP_30d"])
    - df["strike"],
    0,
)  # Calculate the payoff
df["total_payoff_percent_of_liquidity"] = (
    100 * (df["prev_options_minted"] * df["payoff"]) / L
)  # Calculate the total payoff

print("Calculating maximum payoffs...")
non_zero_total_payoff_percent_of_liquidity = df["total_payoff_percent_of_liquidity"][
    df["total_payoff_percent_of_liquidity"] > 0
]  # Filter out zero values
max_total_payoff_percent_of_liquidity = non_zero_total_payoff_percent_of_liquidity.max()
print("Max total payoff percent of liquidity: ", max_total_payoff_percent_of_liquidity)

# Plot 1: Percentage of Cap Reached Over Time with 30-Day TWAP
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(
    df["date"],
    df["total_payoff_percent_of_liquidity"],
    color="b",
    label="Percentage of Cap Reached",
)
ax1.set_title("Percentage Of Cap Reached Over Time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Percentage of Cap Reached (%)", color="b")
ax1.legend(loc="upper left")
ax2 = ax1.twinx()
ax2.plot(df["date"], df["TWAP_30d"], color="r", alpha=0.6, label="30-Day TWAP")
ax2.set_ylabel("30-Day TWAP of Base Fee (ETH)", color="r")
ax2.legend(loc="upper right")
fig.tight_layout()
plt.savefig("plots/extra/cap_reached_vs_twap.png")
plt.close()

# Plot 2: Percentage of Cap Reached Over Time with Cap Level
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(
    df["date"],
    df["total_payoff_percent_of_liquidity"],
    color="b",
    label="Percentage of Cap Reached",
)
ax1.set_title("Percentage Of Cap Reached Over Time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Percentage of Cap Reached (%)", color="b")
ax1.legend(loc="upper left")
ax2 = ax1.twinx()
ax2.plot(
    df["date"], df["cap_level_volatility"], color="r", alpha=0.6, label="Cap Level"
)
ax2.set_ylabel("Cap Level", color="r")
ax2.legend(loc="upper right")
fig.tight_layout()
plt.savefig("plots/extra/cap_reached_vs_cap_level.png")
plt.close()

# Plot 3: Percentage of Cap Reached Over Time with Options Sold
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(
    df["date"],
    df["total_payoff_percent_of_liquidity"],
    color="b",
    label="Percentage of Cap Reached",
)
ax1.set_title("Percentage Of Cap Reached Over Time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Percentage of Cap Reached (%)", color="b")
ax1.legend(loc="upper left")
ax2 = ax1.twinx()
ax2.plot(
    df["date"], df["prev_options_minted"], color="r", alpha=0.6, label="Options Sold"
)
ax2.set_ylabel("Options Sold", color="r")
ax2.legend(loc="upper right")
fig.tight_layout()
plt.savefig("plots/extra/cap_reached_vs_options_sold.png")
plt.close()

print("Testing different k values...")
# Define specific k values to test
k_values = np.array([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
max_payoffs = []

# Loop over each k value with progress bar
for k in tqdm(k_values, desc="Processing k values"):
    # Calculate strike, previous cap level, collateral per option, and options minted for the current k value
    df["strike"] = df["TWAP_30d"].shift(24 * 30) * (1 + k)
    df["previous_cap_level_volatility"] = df["cap_level_volatility"].shift(24 * 30)
    df["previous_collateral_per_option"] = (
        df["previous_cap_level_volatility"] * df["strike"]
    )
    df["prev_options_minted"] = L / df["previous_collateral_per_option"]

    # Calculate payoff and total payoff as a percentage of liquidity
    df["payoff"] = np.maximum(
        np.minimum(
            (1 + df["previous_cap_level_volatility"]) * df["strike"], df["TWAP_30d"]
        )
        - df["strike"],
        0,
    )
    df["total_payoff_percent_of_liquidity"] = (
        100 * (df["prev_options_minted"] * df["payoff"]) / L
    )

    # Filter out zero values
    non_zero_total_payoff_percent_of_liquidity = df[
        "total_payoff_percent_of_liquidity"
    ][df["total_payoff_percent_of_liquidity"] > 0]

    # Record the maximum total payoff percent of liquidity for this k value
    max_total_payoff_percent_of_liquidity = (
        non_zero_total_payoff_percent_of_liquidity.max()
    )
    max_payoffs.append(max_total_payoff_percent_of_liquidity)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(k_values, max_payoffs, marker="o", linestyle="-", color="b")
plt.title("Max Total Payoff Percent of Liquidity vs k")
plt.xlabel("k Value")
plt.ylabel("Max Total Payoff Percent of Liquidity (%)")
plt.grid(True)
plt.savefig("plots/extra/payoff_vs_k.png")
plt.close()

print("\nResults for different k values:")
for k, payoff in zip(k_values, max_payoffs):
    print(f"k = {k:.2f}: {payoff:.2f}%")

print("\nAll plots have been saved to the 'plots' directory.")
print("Done!")


print("Volatility Final Result (data.csv):", df["volatility_90d"].iloc[-1])
print(
    "Volatility Final Result (data.csv) as u128:",
    (10_000 * df["volatility_90d"].iloc[-1]).astype(int),
)
