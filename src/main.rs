mod utilities;
use colored::Colorize; // needed for time

use utilities::tokens::{bench_tonizers, generate_random_tokens};

fn main() {
    let temp_token_file = "_tok.json";
    let num_tokens = 10_000;
    timeit!(generate_random_tokens(temp_token_file, num_tokens).unwrap());
    timeit!(bench_tonizers(temp_token_file).unwrap());
    std::fs::remove_file(temp_token_file).unwrap();
}
