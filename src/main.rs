use anyhow::{anyhow as err, Error};
use argmin::core::Gradient;
use argmin::{
    core::{CostFunction, Executor, observers::ObserverMode},
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS},
};
use argmin_observer_slog::SlogLogger;
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
use argmin_testfunctions::{rosenbrock, rosenbrock_derivative};

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

    df = df
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

    df = df
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

    // let mut to_export = Vec::new()

    // TODO: remove .take(1)
    for (idx, period_df) in dfs.iter().enumerate().take(1) {
        let twap_7d_series = period_df.column("TWAP_7d")?;
        // let strike = twap_7d_series.f64()?.last().ok_or_else(|| err!("The series is empty"));

        // let num_paths = 15000;
        // let n_periods = 720;
        // let cap_level = 0.3;
        // let risk_free_rate = 0.05;

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

        // Optimisation tip: this is same as the time_index, so we can reuse it
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

        // Fitting an ARIMA model to the deseasaonalized and detrended log base fee, finding a distribution for the residuals
        // ===============================================================================================

        // let timeseries = de_seasonalised_detrended_log_base_fee.as_slice().ok_or_else(|| err!("Can't convert to slice"))?;

        // // Fit the model
        // let ar = 12;
        // let d = 0;
        // let ma = 4;
        // let coefficients = fit(timeseries, ar, d, ma)?;

        // // Extract coefficients
        // let intercept = coefficients[0];
        // let phi = &coefficients[1..ar+1];
        // let theta = &coefficients[ar+1..];

        // // Calculate residuals using the provided function
        // let residuals = residuals(&timeseries, intercept, Some(phi), Some(theta))?;

        // println!("Coefficients: {:?}", coefficients);
        // println!("{:?}", Series::new("Residuals", residuals));
        // break;

        // let residuals = df["de_seasonalized_detrended_log_base_fee"].f64()?.to_ndarray()?.to_owned() - Array1::from(arima_fitted_values);
        // let cond_variance = residuals.var(0.0); // ddof needs to match the default value in numpy
        // let standardized_residuals = residuals / cond_variance.sqrt();

        // Monte Carlo Parameter Estimation for the MRJ model
        // ===============================================================================================

        let dt = 1.0 / (365.0 * 24.0);
        let pt = de_seasonalised_detrended_log_base_fee
            .slice(s![1..])
            .to_owned();
        let pt_1 = de_seasonalised_detrended_log_base_fee
            .slice(s![..-1])
            .to_owned();

        let x0 = vec![
            0.0,
            0.0,
            0.7,
            de_seasonalised_detrended_log_base_fee.var(0.0),
            0.1,
            0.005,
        ];

        // println!("X0 {:?}", x0);
        // println!("Dt {:?}", dt);
        // println!("Pt {:?}", pt);
        // println!("Pt_1 {:?}", pt_1);

        // let params_array = Array1::from(x0.clone());
        // let initial_neg_log_likelihood = neg_log_likelihood(&params_array, &pt, &pt_1, dt);
        // println!("Initial negative log-likelihood: {}", initial_neg_log_likelihood);

        // break;
        // let problem = Rosenbrock{};
        let problem = MRJProblem { pt, pt_1, dt };
        let linesearch = MoreThuenteLineSearch::new();
        let solver = LBFGS::new(linesearch, 5);



        let res = Executor::new(problem, solver)
            .configure(|state| state
                .param(Array1::from(x0))
                .max_iters(3)
            )
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()?;

        // Print result
        println!("Optimal parameters: {:?}", res.state().best_param);
        println!("Optimal cost: {}", res.state().best_cost);
    }

    Ok(())
}

fn laplace_pdf(x: f64, mu: f64, b: f64) -> f64 {
    1.0 / (2.0 * b) * (-((x - mu).abs() / b)).exp()
}

fn mrjpdf(params: Array1<f64>, pt: &Array1<f64>, pt_1: &Array1<f64>, dt: f64) -> Array1<f64> {
    // let [a, phi, mu_j, sigma_sq, sigma_sq_j, lambda] = params[..6] else { panic!("Invalid params length") };
    let [a, phi, mu_j, sigma_sq, sigma_sq_j, lambda] = params.slice(s![..6]).to_vec()[..] else {
        panic!("Invalid params length")
    };

    let term1 = lambda
        * pt.iter()
            .zip(pt_1.iter())
            .map(|(p, p1)| {
                laplace_pdf(*p, a * dt + phi * p1 + mu_j, (sigma_sq + sigma_sq_j).sqrt())
            })
            .collect::<Array1<f64>>();

    let term2 = (1.0 - lambda)
        * pt.iter()
            .zip(pt_1.iter())
            .map(|(p, p1)| laplace_pdf(*p, a * dt + phi * p1, sigma_sq.sqrt()))
            .collect::<Array1<f64>>();
    term1 + term2
}

fn neg_log_likelihood(params: Array1<f64>, pt: &Array1<f64>, pt_1: &Array1<f64>, dt: f64) -> f64 {
    let pdf_vals = mrjpdf(params, pt, pt_1, dt);
    
    let log_likelihood: f64 = pdf_vals.mapv(|v| (v + 1e-1).ln()).sum();

    -log_likelihood
}

struct MRJProblem {
    pt: Array1<f64>,
    pt_1: Array1<f64>,
    dt: f64,
}

impl CostFunction for MRJProblem {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let res = neg_log_likelihood(params.clone(), &self.pt, &self.pt_1, self.dt);
        
        println!("Params: {:?}", params);
        println!("Cost fn result: {:?}", res);
        // println!("{:?}", &self.pt);
        // println!("{:?}", &self.pt_1);
        // println!("{:?}", &self.dt);
        println!("\n");

        Ok(res)
    }
}

impl Gradient for MRJProblem {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        // Implement numerical gradient approximation
        // let epsilon = 1e-8;
        // let mut grad = vec![0.0; p.len()];
        // let f0 = self.cost(p)?;

        // for i in 0..p.len() {
        //     let mut p_plus = p.clone();
        //     p_plus[i] += epsilon;
        //     let f_plus = self.cost(&p_plus)?;
        //     grad[i] = (f_plus - f0) / epsilon;
        // }

        // Ok(grad)

        let epsilon = 1e-8;
        let mut grad = Array1::zeros(params.len());
        for i in 0..params.len() {
            let mut params_plus = params.clone();
            params_plus[i] += epsilon;
            let mut params_minus = params.clone();
            params_minus[i] -= epsilon;
            
            let cost_plus = neg_log_likelihood(params_plus, &self.pt, &self.pt_1, self.dt);
            let cost_minus = neg_log_likelihood(
                params_minus, &self.pt, &self.pt_1, self.dt);
            
            grad[i] = (cost_plus - cost_minus) / (2.0 * epsilon);
        }
        Ok(grad)
    }
}

// impl CostFunction for MRJProblem {
//     type Param = Vec<f64>;
//     type Output = f64;

//     fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
//         // Simplified cost function
//         Ok(params.iter().sum())
//     }
// }

// impl Gradient for MRJProblem {
//     type Param = Vec<f64>;
//     type Gradient = Vec<f64>;

//     fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
//         // Simplified gradient function
//         Ok(vec![1.0; p.len()])
//     }
// }

// struct Rosenbrock {}

// impl CostFunction for MRJProblem {
//     type Param = Array1<f64>;
//     type Output = f64;

//     fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//         Ok(rosenbrock(&p.to_vec()))
//     }
// }
// impl Gradient for MRJProblem {
//     type Param = Array1<f64>;
//     type Gradient = Array1<f64>;

//     fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
//         Ok(Array1::from(rosenbrock_derivative(&p.to_vec()).to_vec()))
//     }
// }