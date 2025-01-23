# pitchlake-pricing

### ⚠️⚠️⚠️ This repo is not being maintained ⚠️⚠️⚠️
Here you can find the original Python script and the first Rust implementation.
Latest Rust implementation can be found in the [fossil repo](https://github.com/OilerNetwork/fossil-offchain-processor/tree/main/crates/server/src/pricing_data)

### How to run - Rust impl

1. Unzip the dataset 
`gunzip -c data.csv.gz > data.csv`

2. Build & run the scrip
`cargo build && cargo run`

### How to run - Python impl

Prerequisites: python and poetry

1. Unzip the dataset 
`gunzip -c data.csv.gz > data.csv`

2. Jump into the `python-pricing` directory

3. Install dependencies 
`poetry install`

4. Run the script
`python ReservePricer.py`
