use std::cmp::Ordering;
use std::collections::HashMap;
use std::env::var;
use std::path::Path;

use ndarray::{Array2, ArrayBase, Dim, OwnedRepr};
use onnxruntime::environment::Environment;
use onnxruntime::ndarray::Axis;
use onnxruntime::tensor::ndarray_tensor::NdArrayTensor;
use onnxruntime::{GraphOptimizationLevel, LoggingLevel};

/// Reference used from `NeuML` ;)
/// https://colab.research.google.com/github/neuml/txtai/blob/master/examples/18_Export_and_run_models_with_ONNX.ipynb#scrollTo=_8fdRvO1fFBm
use tokenizers::tokenizer::Tokenizer;
use tokenizers::utils::padding::{
    PaddingDirection::Right, PaddingParams, PaddingStrategy::BatchLongest,
};

use crate::utilities::vec_array::{array2_to_vec, array3_to_vec};

fn tokenize(
    input_texts: &Vec<String>,
    tokenizer_name: &str,
) -> (
    ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
) {
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

    // Encode input text
    let encoding = tokenizer.encode_batch(input_texts.clone(), true).unwrap();

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

#[must_use]
pub fn predict_sentiment(text: &Vec<String>) -> Vec<Vec<f32>> {
    // Start onnx session

    let path = var("RUST_ONNXRUNTIME_LIBRARY_PATH").ok();

    let builder = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Warning);

    let builder = if let Some(path) = path {
        builder.with_library_path(path)
    } else {
        builder
    };

    let environment = builder.build().unwrap();
    // Derive model path
    let model = Path::new("resources/text-classify.onnx");

    let session = environment
        .new_session_builder()
        .unwrap()
        .with_graph_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_model_from_file(model)
        .unwrap();

    let inputs = tokenize(text, "bert-base-uncased");
    let (input_ids, attention_mask, tids) = inputs;

    let outputs = session
        .run(vec![input_ids.into(), attention_mask.into(), tids.into()])
        .unwrap();

    let output = outputs[0].float_array().unwrap();

    array2_to_vec(&output.softmax(Axis(1)).to_owned())
}

#[must_use]
pub fn predict_ner(text: &Vec<String>) -> Vec<Vec<Vec<f32>>> {
    // Start onnx session

    let path = var("RUST_ONNXRUNTIME_LIBRARY_PATH").ok();

    let builder = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Warning);

    let builder = if let Some(path) = path {
        builder.with_library_path(path)
    } else {
        builder
    };

    let environment = builder.build().unwrap();
    // Derive model path
    let model = Path::new("resources/roberta-ner.onnx");

    let session = environment
        .new_session_builder()
        .unwrap()
        .with_graph_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_model_from_file(model)
        .unwrap();

    let inputs = tokenize(text, "xlm-roberta-large-finetuned-conll03-english");
    let (input_ids, attention_mask, _) = inputs;

    let outputs = session
        .run(vec![input_ids.into(), attention_mask.into()])
        .unwrap();

    let output = outputs[0].float_array().unwrap();

    return array3_to_vec(&output.view().to_owned());
}

fn parse_tokens(predictions: &Vec<Vec<Vec<f32>>>) -> Vec<Vec<&str>> {
    let id_labels = HashMap::from([
        (0, "B-LOC"),
        (1, "B-MISC"),
        (2, "B-ORG"),
        (3, "I-LOC"),
        (4, "I-MISC"),
        (5, "I-ORG"),
        (6, "I-PER"),
        (7, "O"),
    ]);

    let res = predictions
        .iter()
        .map(|s| {
            s.iter()
                .map(|t| {
                    id_labels
                        .get(
                            &t.iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| {
                                    a.partial_cmp(b).unwrap_or(Ordering::Equal)
                                })
                                .map(|(index, _)| index)
                                .unwrap(),
                        )
                        .unwrap()
                        .to_owned()
                })
                .collect()
        })
        .collect();

    dbg!(&res);
    res
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    #[test]
    fn test_softmax() {
        // Tokenize input string
        let text_positive = vec!["You are awesome".to_string(), "You are bad".to_string()];

        let responses = predict_sentiment(&text_positive);
        let res_positive = responses.get(0).unwrap();
        let res_negative = responses.get(1).unwrap();

        assert!(res_positive[0] < res_positive[1]);
        println!("{} {:?}", text_positive[0], res_positive);
        assert!(res_negative[0] > res_negative[1]);
    }

    #[test]
    fn test_tokenizer() {
        let test_inputs = vec!["You are awesome".to_string(), "You are bad".to_string()];
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
    #[test]
    fn test_ner() {
        // Tokenize input string
        let text_positive = vec![
            "HuggingFace is a company based in Paris and New York".to_string(),
            "I'm Waner and work for Microsoft from Brazil".to_string(),
        ];

        let responses = predict_ner(&text_positive);
        println!(
            "{:?} {:?} {:?}",
            responses.len(),
            responses[0].len(),
            responses[0][0].len()
        );
        println!("{:?}", parse_tokens(&responses));
    }
}
