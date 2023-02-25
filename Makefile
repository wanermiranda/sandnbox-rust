tidy: 
	cargo fmt
	cargo clippy 

test: 
	cargo test -- \
  --test-threads=1 \
  --nocapture \
  --color=always
	
build: tidy test
	cargo build 

run: tidy
	cargo run
