use chrono::prelude::*;
use linfa::prelude::*;
use linfa::traits::Fit;
use linfa_linear::LinearRegression;
use ndarray::prelude::*;
use polars::prelude::*;

fn read_csv(file: &str) -> PolarsResult<DataFrame> {
    CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(file.into()))?
        .finish()
}

fn main() {
    let data_file = "data.csv";

    let mut df: DataFrame = read_csv(data_file).expect("Cannot read file");

    let dates = df
        .column("timestamp")
        .expect("Cannot find column")
        .i64()
        .expect("Cannot cast to i64")
        .apply(|s| s.map(|s| s * 1000)) // convert into milliseconds
        .into_series()
        .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
        .expect("Cannot cast to datetime");

    df.replace("timestamp", dates)
        .expect("Cannot replace column");
    df.rename("timestamp", "date")
        .expect("Cannot rename column");

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
        .collect()
        .expect("Cannot collect");

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
        .collect()
        .expect("Cannot collect");

    let mut dfs: Vec<DataFrame> = Vec::new();

    let start_date = DateTime::from_timestamp(
        df.column("date")
            .unwrap()
            .datetime()
            .unwrap()
            .get(0)
            .unwrap()
            / 1000,
        0,
    )
    .unwrap();
    let end_date = DateTime::from_timestamp(
        df.column("date")
            .unwrap()
            .datetime()
            .unwrap()
            .get(df.height() - 1)
            .unwrap()
            / 1000,
        0,
    )
    .unwrap();
    let num_months = (end_date.year() - start_date.year()) * 12
        + i32::try_from(end_date.month()).unwrap()
        - i32::try_from(start_date.month()).unwrap()
        + 1;

    for i in 0..num_months - 4 {
        let period_start = start_date + chrono::Months::new(i as u32);
        let period_end = period_start + chrono::Months::new(5);
        let period_df = df
            .clone()
            .lazy()
            .filter(
                col("date")
                    .gt_eq(lit(period_start.naive_utc()))
                    .and(col("date").lt(lit(period_end.naive_utc()))),
            )
            .collect()
            .expect("Cannot collect");
        dfs.push(period_df);
    }

    // let mut to_export = Vec::new()

    for (idx, period_df) in dfs.iter().enumerate() {
        let twap_7d_series = period_df.column("TWAP_7d").expect("Cannot find column");
        let strike = twap_7d_series.f64().unwrap().last().unwrap();

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
            .collect()
            .expect("Cannot collect");

        let log_base_fees: Vec<f64> = df
            .column("base_fee")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .map(|x| x.ln())
            .collect();
        df.with_column(Series::new("log_base_fee", log_base_fees))
            .expect("Cannot add column");

        // Running a linear regression to discover the trend, then removing that trend from the log base fee
        // ===============================================================================================
        // Convert the ndarray to a Series
        let time_index_series = Series::new("time_index", (0..df.height() as i64).into_iter());
        df.with_column(time_index_series)
            .expect("Cannot add column");

        let ones = Array::<f64, Ix1>::ones((df.height() as usize));
        let x = ndarray::stack![
            ndarray::Axis(1),
            Array::from(
                df["time_index"]
                    .cast(&DataType::Float64)
                    .unwrap()
                    .f64()
                    .unwrap()
                    .into_no_null_iter()
                    .collect::<Vec<f64>>()
            ),
            ones
        ];
        let y = Array1::from(
            df["log_base_fee"]
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<f64>>(),
        );

        let dataset = Dataset::<f64, f64, Ix1>::new(x.clone(), y);
        let trend_model = LinearRegression::default()
            .fit(&dataset)
            .expect("Cannot fit model");

        df.with_column(Series::new(
            "trend",
            trend_model.predict(x).targets().to_vec(),
        ))
        .expect("Cannot add column");

        df.with_column(Series::new(
            "detrended_log_base_fee",
            df["log_base_fee"].f64().unwrap() - df["trend"].f64().unwrap(),
        ))
        .expect("Cannot add column");

        // println!("{:?}", df);
        // break;
    }
}
