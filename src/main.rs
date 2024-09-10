use anyhow::{anyhow as err, Error};
use chrono::prelude::*;
use chrono::Months;
use linfa::prelude::*;
use linfa::traits::Fit;
use linfa_linear::LinearRegression;
use ndarray::prelude::*;
use ndarray::{stack, Array1, Array2, Axis};
use ndarray_linalg::LeastSquaresSvd;
use polars::prelude::*;
use std::f64::consts::PI;
use ndarray_rand::rand_distr::Normal;
use rand_distr::Distribution;
use rand::prelude::*;
use statrs::distribution::Binomial;
use optimization::{Func, GradientDescent, Minimizer, NumericalDifferentiation};

#[derive(Debug)]
struct Period {
    starting_timestamp: i64,
    ending_timestamp: i64,
    reserve_price: f64,
    strike_price: f64,
    settlement_price: f64,
    cap_level: i64,
}

fn read_csv(file: &str) -> PolarsResult<DataFrame> {
    CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(file.into()))?
        .finish()
}

fn season_matrix(t: Array1<f64>) -> Array2<f64> {
    let sin_2pi_24 = t.mapv(|time| (2.0 * PI * time / 24.0).sin());
    let cos_2pi_24 = t.mapv(|time| (2.0 * PI * time / 24.0).cos());
    let sin_4pi_24 = t.mapv(|time| (4.0 * PI * time / 24.0).sin());
    let cos_4pi_24 = t.mapv(|time| (4.0 * PI * time / 24.0).cos());
    let sin_8pi_24 = t.mapv(|time| (8.0 * PI * time / 24.0).sin());
    let cos_8pi_24 = t.mapv(|time| (8.0 * PI * time / 24.0).cos());
    let sin_2pi_24_7 = t.mapv(|time| (2.0 * PI * time / (24.0 * 7.0)).sin());
    let cos_2pi_24_7 = t.mapv(|time| (2.0 * PI * time / (24.0 * 7.0)).cos());
    let sin_4pi_24_7 = t.mapv(|time| (4.0 * PI * time / (24.0 * 7.0)).sin());
    let cos_4pi_24_7 = t.mapv(|time| (4.0 * PI * time / (24.0 * 7.0)).cos());
    let sin_8pi_24_7 = t.mapv(|time| (8.0 * PI * time / (24.0 * 7.0)).sin());
    let cos_8pi_24_7 = t.mapv(|time| (8.0 * PI * time / (24.0 * 7.0)).cos());

    stack![
        Axis(1),
        sin_2pi_24,
        cos_2pi_24,
        sin_4pi_24,
        cos_4pi_24,
        sin_8pi_24,
        cos_8pi_24,
        sin_2pi_24_7,
        cos_2pi_24_7,
        sin_4pi_24_7,
        cos_4pi_24_7,
        sin_8pi_24_7,
        cos_8pi_24_7
    ]
}

fn standard_deviation(returns: Vec<f64>) -> f64 {
    let n = returns.len() as f64;
    // Calculate mean
    let mean = returns.iter().sum::<f64>() / n;
    // Calculate sum of squared differences
    let variance = returns.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0); // Use n-1 for sample standard deviation
    // Return the square root of variance
    variance.sqrt()
}

// Function to compute the mean-reverting jump diffusion PDF
fn mrjpdf(params: &[f64], pt: &Array1<f64>, pt_1: &Array1<f64>) -> Array1<f64> {
    let (a, phi, mu_j, sigma_sq, sigma_sq_j, lambda) = (
        params[0], params[1], params[2], params[3], params[4], params[5],
    );

    let term1 = lambda
        * (-((pt - a - phi * pt_1 - mu_j).mapv(|x| x.powi(2))) / (2.0 * (sigma_sq + sigma_sq_j)))
            .mapv(f64::exp)
        / ((2.0 * std::f64::consts::PI * (sigma_sq + sigma_sq_j)).sqrt());

    let term2 = (1.0 - lambda)
        * (-((pt - a - phi * pt_1).mapv(|x| x.powi(2))) / (2.0 * sigma_sq)).mapv(f64::exp)
        / ((2.0 * std::f64::consts::PI * sigma_sq).sqrt());

    term1 + term2
}

// Negative log-likelihood function for the mean-reverting jump diffusion process
fn neg_log_likelihood(params: &[f64], pt: &Array1<f64>, pt_1: &Array1<f64>) -> f64 {
    let pdf_vals = mrjpdf(params, pt, pt_1);
    -pdf_vals.mapv(|x| (x + 1e-10).ln()).sum()
}

fn add_twap_7d(df: DataFrame) -> Result<DataFrame, Error> {
    let df = df
        .lazy()
        .with_column(
            col("base_fee")
                .rolling_mean(RollingOptionsFixedWindow {
                    window_size: 24 * 7,
                    min_periods: 24 * 7,
                    weights: None,
                    center: false,
                    fn_params: None,
                })
                .alias("TWAP_7d"),
        )
        .collect()?;

    Ok(df)
}

fn group_by_1h_intervals(df: DataFrame) -> Result<DataFrame, Error> {
    let df = df
        .lazy()
        .group_by_dynamic(
            col("date"),
            [],
            DynamicGroupOptions {
                every: Duration::parse("1h"),
                period: Duration::parse("1h"),
                offset: Duration::parse("0"),
                ..Default::default()
            },
        )
        .agg([
            col("base_fee").mean(),
            col("gas_limit").mean(),
            col("gas_used").mean(),
            col("number").mean(),
        ])
        .collect()?;

    Ok(df)
}

fn main() -> Result<(), Error> {
    let data_file = "data.csv";

    let mut df: DataFrame = read_csv(data_file).expect("Cannot read file");

    let dates = df
        .column("timestamp")?
        .i64()?
        .apply(|s| s.map(|s| s * 1000)) // convert into milliseconds
        .into_series()
        .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?;

    df.replace("timestamp", dates)?;
    df.rename("timestamp", "date")?;

    df = group_by_1h_intervals(df)?;
    df = add_twap_7d(df)?;

    let mut dfs: Vec<DataFrame> = Vec::new();

    let start_date_value = df
        .column("date")?
        .datetime()?
        .get(0)
        .ok_or_else(|| err!("No row 0 in the date column"))?;
    let start_date = DateTime::from_timestamp(start_date_value / 1000, 0)
        .ok_or_else(|| err!("Can't calculate the start date"))?;

    let end_date_row = df.height() - 1;
    let end_date_value = df
        .column("date")?
        .datetime()?
        .get(end_date_row)
        .ok_or_else(|| err!("No row {end_date_row} in the date column"))?;
    let end_date = DateTime::from_timestamp(end_date_value / 1000, 0)
        .ok_or_else(|| err!("Can't calculate the end date"))?;

    let num_months = (end_date.year() - start_date.year()) * 12 + i32::try_from(end_date.month())?
        - i32::try_from(start_date.month())?
        + 1;

    for i in 0..num_months - 4 {
        let period_start = start_date + Months::new(i as u32);
        let period_end = period_start + Months::new(5);
        let period_df = df
            .clone()
            .lazy()
            .filter(
                col("date")
                    .gt_eq(lit(period_start.naive_utc()))
                    .and(col("date").lt(lit(period_end.naive_utc()))),
            )
            .collect()?;

        dfs.push(period_df);
    }

    let mut to_export = Vec::<Period>::new();

    for (idx, period_df) in dfs.iter().enumerate() {
        let twap_7d_series = period_df.column("TWAP_7d")?;
        let strike = twap_7d_series
            .f64()?
            .last()
            .ok_or_else(|| err!("The series is empty"))?;

        let num_paths = 15000;
        let n_periods = 720;
        let cap_level = 0.3;
        let risk_free_rate = 0.05;

        // Data Cleaning and Preprocessing - removing null if exist and log transformation
        // ===============================================================================================

        // drop rows with null values
        let mut df = period_df
            .clone()
            .lazy()
            .filter(col("TWAP_7d").is_not_null())
            .collect()?;

        let log_base_fees: Vec<f64> = df
            .column("base_fee")?
            .f64()?
            .into_no_null_iter()
            .map(|x| x.ln())
            .collect();
        df.with_column(Series::new("log_base_fee", log_base_fees))?;

        // Running a linear regression to discover the trend, then removing that trend from the log base fee
        // ===============================================================================================

        let time_index_series = Series::new("time_index", (0..df.height() as i64).into_iter());
        df.with_column(time_index_series)?;

        let ones = Array::<f64, Ix1>::ones(df.height() as usize);
        let x = stack![
            Axis(1),
            Array::from(
                df["time_index"]
                    .cast(&DataType::Float64)?
                    .f64()?
                    .into_no_null_iter()
                    .collect::<Vec<f64>>()
            ),
            ones
        ];

        let y = Array1::from(
            df["log_base_fee"]
                .f64()?
                .into_no_null_iter()
                .collect::<Vec<f64>>(),
        );

        let dataset = Dataset::<f64, f64, Ix1>::new(x.clone(), y);
        let trend_model = LinearRegression::default()
            .with_intercept(false)
            .fit(&dataset)?;

        df.with_column(Series::new(
            "trend",
            trend_model.predict(x).targets().to_vec(),
        ))?;

        df.with_column(Series::new(
            "detrended_log_base_fee",
            df["log_base_fee"].f64()? - df["trend"].f64()?,
        ))?;

        // Seasonality modelling amd removal from the detrended log base fee
        // ===============================================================================================

        let start_date_value = df
            .column("date")?
            .datetime()?
            .get(0)
            .ok_or_else(|| err!("No row 0 in the date column"))?;
        let start_date = DateTime::from_timestamp(start_date_value / 1000, 0)
            .ok_or_else(|| err!("Can't calculate the start date"))?;

        let t_series: Vec<f64> = df
            .column("date")?
            .datetime()?
            .into_iter()
            .map(|opt_date| {
                opt_date.map_or(0.0, |date| {
                    (DateTime::from_timestamp(date / 1000, 0).unwrap() - start_date).num_seconds()
                        as f64
                        / 3600.0
                })
            })
            .collect();

        df.with_column(Series::new("t", t_series))?;

        let t_array = df["t"].f64()?.to_ndarray()?.to_owned();
        let c = season_matrix(t_array);

        let detrended_log_base_fee_array =
            df["detrended_log_base_fee"].f64()?.to_ndarray()?.to_owned();
        let season_param = c.least_squares(&detrended_log_base_fee_array)?.solution;
        let season = c.dot(&season_param);
        let de_seasonalised_detrended_log_base_fee =
            df["detrended_log_base_fee"].f64()?.to_ndarray()?.to_owned() - season;
        df.with_column(Series::new(
            "de_seasonalized_detrended_log_base_fee",
            de_seasonalised_detrended_log_base_fee.clone().to_vec(),
        ))?;

        // Monte Carlo Parameter Estimation for the MRJ model
        // ===============================================================================================

        let dt = 1.0 / (365.0 * 24.0);
        let pt = de_seasonalised_detrended_log_base_fee
            .slice(s![1..])
            .to_owned();
        let pt_1 = de_seasonalised_detrended_log_base_fee
            .slice(s![..-1])
            .to_owned();

        let function =
            NumericalDifferentiation::new(Func(|x: &[f64]| neg_log_likelihood(x, &pt, &pt_1)));

        let minimizer = GradientDescent::new().max_iterations(Some(300));

        let var_pt = pt.var(0.0); // The 0.0 here refers to the degrees of freedom, equivalent to ddof in numpy
        let solution = minimizer.minimize(
            &function,
            vec![-3.928e-02, 2.873e-04, 4.617e-02, var_pt, var_pt, 0.2],
        );

        let params = &solution.position; // Get the optimized parameters
        let alpha = params[0] / dt;
        let kappa = (1.0 - params[1]) / dt;
        let mu_j = params[2];
        let sigma = (params[3] / dt).sqrt();
        let sigma_j = params[4].sqrt();
        let lambda_ = params[5] / dt;

        // println!("Found solution for Rosenbrock function at f({:?}) = {:?}",
        //     solution.position, solution.value);
        // println!("Fitted params: {:?}", params);
        // println!("alpha: {}", alpha);
        // println!("kappa: {}", kappa);
        // println!("mu_J: {}", mu_j);
        // println!("sigma: {}", sigma);
        // println!("sigma_J: {}", sigma_j);
        // println!("lambda_: {}", lambda_);

        // Monte Carlo Simulation of the MRJ model
        // ===============================================================================================

        let mut rng = thread_rng();
        let j: Array2<f64> = {
            let binom = Binomial::new(lambda_ * dt, 1)?;
            Array2::from_shape_fn((n_periods, num_paths), |_| binom.sample(&mut rng) as f64)
        };

        let mut sim_prices = Array2::zeros((n_periods, num_paths));
        sim_prices.slice_mut(s![0, ..]).assign(&Array1::from_elem(
            num_paths,
            de_seasonalised_detrended_log_base_fee
                [de_seasonalised_detrended_log_base_fee.len() - 1],
        ));

        let normal = Normal::new(0.0, 1.0).unwrap();
        let n1 = Array2::from_shape_fn((n_periods, num_paths), |_| normal.sample(&mut rng));
        let n2 = Array2::from_shape_fn((n_periods, num_paths), |_| normal.sample(&mut rng));

        // Simulate the prices over time
        for i in 1..n_periods {
            let prev_prices = sim_prices.slice(s![i - 1, ..]);
            let current_n1 = n1.slice(s![i, ..]);
            let current_n2 = n2.slice(s![i, ..]);
            let current_j = j.slice(s![i, ..]);

            let new_prices = &(alpha * dt
                + (1.0 - kappa * dt) * &prev_prices
                + sigma * dt.sqrt() * &current_n1
                + &current_j * (mu_j + sigma_j * &current_n2));

            sim_prices.slice_mut(s![i, ..]).assign(&new_prices);
        }

        let last_date_value = df
            .column("date")?
            .datetime()?
            .get(df.height() - 1)
            .ok_or_else(|| err!("No row {end_date_row} in the date column"))?;

        let start_date_value = df
            .column("date")?
            .datetime()?
            .get(0)
            .ok_or_else(|| err!("No row 0 in the date column"))?;

        // Calculate the total hours between start and last date
        let total_hours = (last_date_value - start_date_value) / 3600 / 1000;

        // Generate an array of elapsed hours
        let sim_hourly_times: Array1<f64> =
            Array1::range(0.0, n_periods as f64, 1.0).mapv(|i| total_hours as f64 + i);

        // Adding seasonality back to the simulated prices
        // ===============================================================================================
        let c = season_matrix(sim_hourly_times);
        let season = c.dot(&season_param);

        // Reshape season to (n_periods, 1)
        let season_reshaped = season.into_shape((n_periods, 1)).unwrap();

        // Broadcasting addition of season to sim_prices
        let log_sim_prices = &sim_prices + &season_reshaped;

        //  Calibrating and adding stochastic trend to the simulation.
        //  ===============================================================================================

        let log_twap_7d: Vec<f64> = df
            .column("TWAP_7d")?
            .f64()?
            .into_no_null_iter()
            .map(|x| x.ln())
            .collect();

        // Compute the difference between consecutive elements in log_twap_7d
        let returns: Vec<f64> = log_twap_7d
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect();

        // Drop NaNs from returns
        let returns: Vec<f64> = returns.into_iter().filter(|&x| !x.is_nan()).collect();

        let mu = 0.05 / 52.0; // Weekly drift
        let sigma = standard_deviation(returns) * f64::sqrt(24.0 * 7.0); // Weekly voldatility
        let dt = 1.0 / 24.0;

        let mut stochastic_trend = Array2::<f64>::zeros((n_periods, num_paths));

        // Generate random shocks for each path
        let normal = Normal::new(0.0, sigma * (f64::sqrt(dt))).unwrap();
        for i in 0..num_paths {
            let random_shocks: Vec<f64> = (0..n_periods).map(|_| normal.sample(&mut rng)).collect();

            // Calculate cumulative sum for stochastic trend
            let mut cumsum = 0.0;
            for j in 0..n_periods {
                cumsum += (mu - 0.5 * sigma.powi(2)) * dt + random_shocks[j];
                stochastic_trend[[j, i]] = cumsum;
            }
        }

        // Adding trend and stochastic trend to the simulation, considering the final trend value
        // =================================================

        let coeffs = trend_model.params();
        let final_trend_value = {
            let x = (df.height() - 1) as f64;
            coeffs[0] * x + coeffs[1]
        };

        // Use final_trend_value in your simulation
        let mut sim_log_prices_with_trend = Array2::<f64>::zeros((n_periods, num_paths));
        for i in 0..n_periods {
            let trend = final_trend_value; // Use the final trend value for all future time points
            for j in 0..num_paths {
                sim_log_prices_with_trend[[i, j]] =
                    log_sim_prices[[i, j]] + trend + stochastic_trend[[i, j]];
            }
        }

        // Convert log prices to actual prices
        let sim_prices = sim_log_prices_with_trend.mapv(f64::exp);

        // Calculate TWAP
        let twap_start = n_periods.saturating_sub(24 * 7);
        let final_prices_twap = sim_prices
            .slice(s![twap_start.., ..])
            .mean_axis(Axis(0))
            .unwrap();

        let payoffs = final_prices_twap.mapv(|price| {
            let capped_price = (1.0 + cap_level) * strike;
            let payoff = (price.min(capped_price) - strike).max(0.0);
            payoff
        });

        let average_payoff = payoffs.mean().unwrap_or(0.0);
        let present_value = f64::exp(-risk_free_rate) * average_payoff;

        let mut settlement_price = 0.0;
        let mut ending_timestamp = 0;

        if idx + 1 < dfs.len() {
            let next_period_df = dfs[idx + 1].clone();
            settlement_price = next_period_df.column("TWAP_7d")?.f64()?.last().unwrap();
            ending_timestamp = next_period_df.column("date")?.datetime()?.last().unwrap();
        }

        to_export.push(Period {
            starting_timestamp: period_df.column("date")?.datetime()?.get(0).unwrap(),
            ending_timestamp: ending_timestamp,
            reserve_price: present_value,
            strike_price: strike,
            settlement_price: settlement_price, // This will be filled later
            cap_level: (cap_level * 10000.0) as i64, // in basis points
        });
    }

    println!("Output:\n{:?}", to_export);
    Ok(())
}
