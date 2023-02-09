use tokenizers::models::wordpiece::WordPiece;
use tokenizers::{AddedToken, Result, Tokenizer};

/// It loads a tokenizer from a file
///
/// Arguments:
///
/// * `output_path`: The path to the output file.
///
/// Returns:
///
/// A Tokenizer
pub fn bench_tonizers(output_path: &str) -> Result<Tokenizer> {
    Tokenizer::from_file(output_path)
}

/// It creates a tokenizer, adds some tokens to it, and saves it to disk
///
/// A doc comment.
/// Arguments:
///
/// * `output_path`: The path to save the tokenizer to.
/// * `num_tokens`: The number of tokens to generate.
///
/// Returns:
///
/// A Result<()>
///
/// # Examples:
///
/// ```    
///use sandbox_rust::utilities::tokens::generate_random_tokens;
///let temp_token_file = "test_tok.json";
///let num_tokens = 100;
///generate_random_tokens(temp_token_file, num_tokens).unwrap();
///std::fs::remove_file(temp_token_file).unwrap();
//    ```
pub fn generate_random_tokens(output_path: &str, num_tokens: u32) -> Result<()> {
    let mut tokenizer = Tokenizer::new(WordPiece::default());

    // Mix special and not special
    // You can make sure ids are in order, and special status is correct.
    let tokens: Vec<_> = (0..num_tokens)
        .map(|i| AddedToken::from(format!("[SPECIAL_{}]", i), i % 2 == 0))
        .collect();
    tokenizer.add_tokens(&tokens);
    tokenizer.save(output_path, true)
}
