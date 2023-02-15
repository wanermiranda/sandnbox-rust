use rust_tokenizers::{error::TokenizerError, tokenizer::RobertaTokenizer};

use cached_path::{Cache, Error, ProgressBar};
use std::path::PathBuf;

pub fn download_file_to_cache(src: &str) -> Result<PathBuf, Error> {
    let mut cache_dir = dirs::home_dir().unwrap();
    cache_dir.push(".cache");
    cache_dir.push(".rust_tokenizers");

    let cached_path = Cache::builder()
        .dir(cache_dir)
        .progress_bar(Some(ProgressBar::Light))
        .build()?
        .cached_path(src)?;
    Ok(cached_path)
}

// Define a struct for the RoBERTa tokenizer
// Initialize the RoBERTa tokenizer
/// We load the tokenizer from the files `roberta-base-tokenizer.json` and
/// `roberta-base-sentencepiece.bpe.model` and we set the `lower_case` and `add_prefix_space` parameters
/// to `false` and `true` respectively
///
/// Returns:
///
/// A RobertaTokenizer
///
/// example:
///
/// ```
/// let tokenenizer = build_tokenizer().unwrap();
/// let text_list = = [
///        "My name is Amélie. I live in Москва.",
///        "Chongqing is a city in China.",
///        "Meu nome é Waner e moro no Brasil.",
///        "My name is Mario and I live in Canada.",
///    ];
/// tokenizer.tokenize_list(text_list);
/// ```
pub fn build_tokenizer() -> Result<RobertaTokenizer, TokenizerError> {
    let lower_case = false;
    let add_prefix_space = true;

    let vocab_path = download_file_to_cache(
        "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
    )
    .unwrap();

    let merges_path = download_file_to_cache(
        "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
    )
    .unwrap();

    let tokenizer = RobertaTokenizer::from_file(
        vocab_path,  // "/home/gorigan/projects/sandbox-rust/resources/roberta-base-vocab.json",
        merges_path, // "/home/gorigan/projects/sandbox-rust/resources/roberta-base-sentencepiece.bpe.model",
        lower_case,
        add_prefix_space,
    )
    .unwrap();

    Ok(tokenizer)
}
