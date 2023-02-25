use ndarray::{Array2, ArrayBase, Dim, OwnedRepr};
use tokenizers::tokenizer::Tokenizer;
use tokenizers::utils::padding::{
    PaddingDirection::Right, PaddingParams, PaddingStrategy::BatchLongest,
};

pub type Embeddings = (
    ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
);

pub fn tokenize<S>(input_texts: &[S], tokenizer_name: &str) -> Embeddings
where
    S: AsRef<str>,
{
    let batch_size = input_texts.len();
    // Load tokenizer from HF Hub
    let mut tokenizer = Tokenizer::from_pretrained(tokenizer_name, None).unwrap();

    tokenizer.with_padding(Some(PaddingParams {
        strategy: BatchLongest,
        direction: Right,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]".into(),
        pad_to_multiple_of: Some(2),
    }));

    let inputs = input_texts.iter().map(|s| s.as_ref()).collect();

    // Encode input text
    let encoding = tokenizer.encode_batch(inputs, true).unwrap();

    let max_len = encoding
        .iter()
        .map(|feature| feature.get_ids().len())
        .max()
        .unwrap();

    let input_shape = (batch_size, max_len);

    let mut masks = Array2::<i64>::zeros(input_shape);
    let mut token_ids = Array2::<i64>::zeros(input_shape);
    let mut type_ids = Array2::<i64>::zeros(input_shape);

    for (i, e) in encoding.iter().enumerate() {
        for j in 0..max_len {
            token_ids[[i, j]] = i64::from(e.get_ids()[j].to_owned());
            masks[[i, j]] = i64::from(e.get_attention_mask()[j].to_owned());
            type_ids[[i, j]] = i64::from(e.get_type_ids()[j].to_owned());
        }
    }

    (token_ids, masks, type_ids)
}

mod tests {
    use ndarray::array;

    use super::*;
    #[test]
    fn test_tokenizer() {
        let test_inputs = vec!["You are awesome", "You are bad"];
        let token_results = tokenize(&test_inputs, "bert-base-uncased");
        let (input_ids, attention_mask, tids) = token_results;
        println!("{input_ids:?}");
        assert_eq!(
            array![
                [101, 2017, 2024, 12476, 102, 0],
                [101, 2017, 2024, 2919, 102, 0]
            ],
            input_ids
        );

        println!("{attention_mask:?}");
        assert_eq!(
            array![[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]],
            attention_mask
        );

        println!("{tids:?}");
        assert_eq!(array![[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], tids);
    }
}
