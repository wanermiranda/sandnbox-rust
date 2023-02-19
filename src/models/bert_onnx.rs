use std::env::var;
use std::path::Path;

use ndarray::{arr2, Array, Array2, ArrayBase, Dim, OwnedRepr};
use onnxruntime::environment::Environment;
use onnxruntime::ndarray::Axis;
use onnxruntime::tensor::ndarray_tensor::NdArrayTensor;
use onnxruntime::{GraphOptimizationLevel, LoggingLevel};

/// Reference used from NeuML ;)
/// https://colab.research.google.com/github/neuml/txtai/blob/master/examples/18_Export_and_run_models_with_ONNX.ipynb#scrollTo=_8fdRvO1fFBm
use tokenizers::tokenizer::Tokenizer;
use tokenizers::utils::padding::{
    PaddingDirection::Right, PaddingParams, PaddingStrategy::BatchLongest,
};

fn tokenize(
    input_texts: &Vec<String>,
) -> (
    ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
) {
    let batch_size = input_texts.len();
    // Load tokenizer from HF Hub
    let mut tokenizer = Tokenizer::from_pretrained("bert-base-uncased", None).unwrap();

    tokenizer.with_padding(Some(PaddingParams {
        strategy: BatchLongest,
        direction: Right,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]".into(),
        pad_to_multiple_of: todo!(),
    }));

    // Encode input text
    let encoding = tokenizer
        .encode_batch(input_texts.to_owned(), true)
        .unwrap();

    let v1: Vec<Vec<i64>> = encoding
        .iter()
        .map(|e| {
            e.get_ids()
                .to_vec()
                .iter()
                .map(|x| x.to_owned() as i64)
                .collect()
        })
        .collect();

    let v2: Vec<Vec<i64>> = encoding
        .iter()
        .map(|e| {
            arr2(e.get_attention_mask())
                .map(|x| x.to_owned() as i64)
                .collect()
        })
        .collect();

    let v3: Vec<Vec<i64>> = encoding
        .iter()
        .map(|e| {
            e.get_type_ids()
                .to_vec()
                .iter()
                .map(|x| x.to_owned() as i64)
                .collect()
        })
        .collect();

    let ids = Array::from_shape_vec((batch_size, v1.len()), v1).unwrap();
    let mask = Array::from_shape_vec((batch_size, v2.len()), v2).unwrap();
    let tids = Array::from_shape_vec((batch_size, v3.len()), v3).unwrap();

    (ids, mask, tids)
}

pub fn predict(text: &Vec<String>, softmax: bool) -> Vec<f32> {
    // Start onnx session

    let path = var("RUST_ONNXRUNTIME_LIBRARY_PATH").ok();

    let builder = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Warning);

    let builder = if let Some(path) = path.clone() {
        builder.with_library_path(path)
    } else {
        builder
    };

    let environment = builder.build().unwrap();
    // Derive model path
    let model = if softmax {
        Path::new("resources/text-classify.onnx")
    } else {
        Path::new("resources/roberta-ner.onnx")
    };

    let session = environment
        .new_session_builder()
        .unwrap()
        .with_graph_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_model_from_file(model)
        .unwrap();

    let inputs = tokenize(text);
    let (input_ids, attention_mask, tids) = inputs;

    let outputs = session
        .run(vec![input_ids.into(), attention_mask.into(), tids.into()])
        .unwrap();

    let output = outputs[0].float_array().unwrap();

    if softmax {
        return output
            .softmax(Axis(1))
            .iter()
            .map(|y| y.to_owned())
            .collect::<Vec<_>>();
    } else {
        return output.iter().map(|y| y.to_owned()).collect::<Vec<_>>();
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    // #[test]
    // fn test_softmax() {
    //     // Tokenize input string
    //     let text_positive = "You are awesome".to_string();

    //     let res_positive = predict(&text_positive, true);
    //     assert!(res_positive[0] < res_positive[1]);
    //     println!("{} {:?}", text_positive, res_positive);

    //     let text_negative = "You are bad".to_string();

    //     let res_negative = predict(&text_negative, true);
    //     assert!(res_negative[0] > res_negative[1]);
    //     println!("{} {:?}", text_negative, res_negative);
    // }

    #[test]
    fn test_tokenizer() {
        let test_inputs = vec!["You are awesome".to_string(), "You are bad".to_string()];
        let token_results = tokenize(&test_inputs);
        let (input_ids, attention_mask, tids) = token_results;
        println!("{:?}", input_ids);
    }

    // fn test_ner() {
    //     // Tokenize input string
    //     let text_positive = "You are awesome".to_string();

    //     let res_positive = predict(&text_positive, true);
    //     assert!(res_positive[0] < res_positive[1]);
    //     println!("{} {:?}", text_positive, res_positive);

    //     let text_negative = "You are bad".to_string();

    //     let res_negative = predict(&text_negative, true);
    //     assert!(res_negative[0] > res_negative[1]);
    //     println!("{} {:?}", text_negative, res_negative);
    // }
}
