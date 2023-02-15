use std::env::var;
use std::path::Path;

use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use onnxruntime::environment::Environment;
use onnxruntime::ndarray::Axis;
use onnxruntime::tensor::ndarray_tensor::NdArrayTensor;
use onnxruntime::{GraphOptimizationLevel, LoggingLevel};

/// Reference used from NeuML ;)
/// https://colab.research.google.com/github/neuml/txtai/blob/master/examples/18_Export_and_run_models_with_ONNX.ipynb#scrollTo=_8fdRvO1fFBm
use tokenizers::tokenizer::Tokenizer;

fn tokenize(
    text: &String,
    inputs: usize,
) -> (
    ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
) {
    // Load tokenizer from HF Hub
    let tokenizer = Tokenizer::from_pretrained("bert-base-uncased", None).unwrap();

    // Encode input text
    let encoding = tokenizer.encode(text.to_owned(), true).unwrap();

    let v1: Vec<i64> = encoding
        .get_ids()
        .to_vec()
        .into_iter()
        .map(|x| x as i64)
        .collect();
    let v2: Vec<i64> = encoding
        .get_attention_mask()
        .to_vec()
        .into_iter()
        .map(|x| x as i64)
        .collect();
    let v3: Vec<i64> = encoding
        .get_type_ids()
        .to_vec()
        .into_iter()
        .map(|x| x as i64)
        .collect();

    let ids = Array::from_shape_vec((1, v1.len()), v1).unwrap();
    let mask = Array::from_shape_vec((1, v2.len()), v2).unwrap();
    let tids = Array::from_shape_vec((1, v3.len()), v3).unwrap();

    (ids, mask, tids)
}

fn predict(text: &String, softmax: bool) -> Vec<f32> {
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
        Path::new("resources/embeddings.onnx")
    };

    let session = environment
        .new_session_builder()
        .unwrap()
        .with_graph_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_model_from_file(model)
        .unwrap();

    let inputs = tokenize(text, session.inputs.len());
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
    #[test]
    fn test_onnx_softmax() {
        // Tokenize input string
        let text_positive = "You are awesome".to_string();

        let res_positive = predict(&text_positive, true);
        assert!(res_positive[0] < res_positive[1]);
        println!("{} {:?}", text_positive, res_positive);

        let text_negative = "You are bad".to_string();

        let res_negative = predict(&text_negative, true);
        assert!(res_negative[0] > res_negative[1]);
        println!("{} {:?}", text_negative, res_negative);
    }
}
