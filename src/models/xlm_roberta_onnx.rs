use std::cmp::Ordering;
use std::collections::HashMap;
use std::env::var;
use std::path::Path;

use crate::tokens::bert_roberta_tokenizers::tokenize;
use crate::utilities::vec_array::{array2_to_vec, array3_to_vec};

use onnxruntime::environment::Environment;
use onnxruntime::ndarray::Axis;
use onnxruntime::session::Session;
use onnxruntime::tensor::ndarray_tensor::NdArrayTensor;
use onnxruntime::{GraphOptimizationLevel, LoggingLevel};

/// Reference used from `NeuML` ;)
/// https://colab.research.google.com/github/neuml/txtai/blob/master/examples/18_Export_and_run_models_with_ONNX.ipynb#scrollTo=_8fdRvO1fFBm

pub fn predict_sentiment(text: &[&str]) -> Vec<Vec<f32>> {
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

pub fn predict<S>(text: &[S], session: &Session) -> Vec<Vec<Vec<f32>>>
where
    S: AsRef<str>,
{
    // Start onnx session

    let inputs = tokenize(text, "xlm-roberta-large-finetuned-conll03-english");
    let (input_ids, attention_mask, _) = inputs;

    let outputs = session
        .run(vec![input_ids.into(), attention_mask.into()])
        .unwrap();

    let output = outputs[0].float_array().unwrap();

    return array3_to_vec(&output.view().to_owned());
}

pub fn build_model() -> Session {
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
    session
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
    use super::*;
    #[test]
    fn test_softmax() {
        // Tokenize input string
        let text_positive = vec!["You are awesome", "You are bad"];

        let responses = predict_sentiment(&text_positive);
        let res_positive = responses.get(0).unwrap();
        let res_negative = responses.get(1).unwrap();

        assert!(res_positive[0] < res_positive[1]);
        println!("{} {:?}", text_positive[0], res_positive);
        assert!(res_negative[0] > res_negative[1]);
    }

    #[test]
    fn test_ner() {
        // Tokenize input string
        let text_positive = [
            "HuggingFace is a company based in Paris and New York",
            "I'm Waner and work for Microsoft from Brazil",
        ];
        let session = build_model();

        let responses = predict(&text_positive, &session);
        println!(
            "{:?} {:?} {:?}",
            responses.len(),
            responses[0].len(),
            responses[0][0].len()
        );
        println!("{:?}", parse_tokens(&responses));
    }
}
