#[allow(unused_imports)]
use rust_tokenizers::{
    error::TokenizerError,
    tokenizer::{MultiThreadedTokenizer, RobertaTokenizer},
};

use crate::utilities::files::download_file_to_cache;

// Define a struct for the RoBERTa tokenizer
// Initialize the RoBERTa tokenizer
/// We load the tokenizer from the files `roberta-base-tokenizer.json` and
/// `roberta-base-sentencepiece.bpe.model` and we set the `lower_case` and `add_prefix_space` parameters
/// to `false` and `true` respectively
///
/// Returns:
///
/// A `RobertaTokenizer`
///
#[allow(dead_code)]
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
        vocab_path,  // "resources/roberta-base-vocab.json",
        merges_path, // "resources/roberta-base-sentencepiece.bpe.model",
        lower_case,
        add_prefix_space,
    )
    .unwrap();

    Ok(tokenizer)
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_roberta_tokenizer_rustbert() {
        let ltokenenizer = build_tokenizer().unwrap();
        let text_list = [
            "My name is Amélie. I live in Москва.",
            "Chongqing is a city in China.",
            "Meu nome é Waner e moro no Brasil.",
            "My name is Mario and I live in Canada.",
        ];
        ltokenenizer.tokenize_list(&text_list);
    }
}
