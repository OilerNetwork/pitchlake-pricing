use polars_core::prelude::*;
use polars_io::prelude::*;

fn read_csv(file: &str) -> PolarsResult<DataFrame> {
    CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(file.into()))?
            .finish()
}

fn main() {
    let data_file = "output.csv";

    let df : PolarsResult<DataFrame> = read_csv(data_file);
    println!("{:?}", df);
}
