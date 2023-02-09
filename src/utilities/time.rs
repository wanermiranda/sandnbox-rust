/// A macro that takes an expression and prints out the time it took to run the expression.
#[macro_export]
macro_rules! timeit {
    ($expression:expr) => {
        let start = std::time::Instant::now();
        let result = $expression;

        "this is blue".blue();

        println!(
            "Command: \n\t {} \n\t took {}",
            format!("{}", stringify!($expression).bold().blue()),
            format!("{:?}", start.elapsed()).italic()
        );
        drop(result)
    };
}
