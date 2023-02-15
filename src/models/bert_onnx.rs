use onnxruntime::environment::Environment;
use onnxruntime::ndarray::{Array2, Axis};
use onnxruntime::tensor::OrtOwnedTensor;
use onnxruntime::GraphOptimizationLevel;
/// Reference used from NeuML ;)
/// https://colab.research.google.com/github/neuml/txtai/blob/master/examples/18_Export_and_run_models_with_ONNX.ipynb#scrollTo=_8fdRvO1fFBm
use std::env;

use tokenizers::tokenizer::{Result, Tokenizer};

fn tokenize(text: String, inputs: usize) -> Vec<Array2<i64>> {
    // Load tokenizer from HF Hub
    let tokenizer = Tokenizer::from_pretrained("bert-base-uncased", None).unwrap();

    // Encode input text
    let encoding = tokenizer.encode(text, true).unwrap();

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

    let ids = Array2::from_shape_vec((1, v1.len()), v1).unwrap();
    let mask = Array2::from_shape_vec((1, v2.len()), v2).unwrap();
    let tids = Array2::from_shape_vec((1, v3.len()), v3).unwrap();

    return if inputs > 2 {
        vec![ids, mask, tids]
    } else {
        vec![ids, mask]
    };
}

fn predict(text: String, softmax: bool) -> Vec<f32> {
    // Start onnx session
    let environment = Environment::builder().with_name("test").build().unwrap();

    // Derive model path
    let model = if softmax {
        "text-classify.onnx"
    } else {
        "embeddings.onnx"
    };

    let mut session = environment
        .new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_number_threads(1)
        .unwrap()
        .with_model_from_file(model)
        .unwrap();

    let inputs = tokenize(text, session.inputs.len());

    // Run inference and print result
    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(inputs).unwrap();
    let output: &OrtOwnedTensor<f32, _> = &outputs[0];

    let probabilities: Vec<f32>;
    if softmax {
        probabilities = output.softmax(Axis(1)).iter().copied().collect::<Vec<_>>();
    } else {
        probabilities = output.iter().copied().collect::<Vec<_>>();
    }

    return probabilities;
}

fn test_onnx(test_str: String) -> Result<()> {
    // Tokenize input string
    let args: Vec<String> = env::args().collect();

    let v1 = predict(args[1].to_string(), true);
    println!("{:?}", v1);

    Ok(())
}
