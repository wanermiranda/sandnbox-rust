tidy: 
	cargo fmt
	cargo clippy --fix --allow-dirty -- -W clippy::pedantic -W clippy::nursery -W clippy::unwrap_used -W clippy::expect_used 

test: 
	cargo test -- \
  --test-threads=1 \
  --nocapture \
  --color=always
	
build: tidy test
	cargo build 

run: tidy
	cargo run
