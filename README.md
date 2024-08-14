# pitchlake-pricing

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
