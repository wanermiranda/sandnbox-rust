let builder = environment::Environment::builder()
        .with_name("onnx_metadata")
        .with_log_level(LoggingLevel::Verbose);

    let environment = builder.build().unwrap();

    // provide path to .onnx model on disk
    let path: String = String::from(
        "/home/gorigan/projects/sandbox-rust/resources/roberta-sequence-classification-9.onnx",
    );

    let session = environment
        .new_session_builder()
        .unwrap()
        .with_model_from_file(path)
        .unwrap();

    let tokenizer = roberta_tokenizer::build_tokenizer().unwrap();

    let input = [
        "My name is Amélie. I live in Москва.",
        "Chongqing is a city in China.",
        "Meu nome é Waner e moro no Brasil.",
        "My name is Mario and I live in Canada.",
    ];

    let features = tokenizer.encode_list(&input, 128, &TruncationStrategy::LongestFirst, 0);

    let max_len = features
        .iter()
        .map(|feature| feature.token_ids.len())
        .max()
        .unwrap();

    let input_shape = (features.len(), max_len);

    let input_ids = ndarray::Array::from_vec(features.iter().map(|i| i.token_ids).collect());
    // .into_shape(input_shape)
    // .unwrap();

    let attention_masks = ndarray::Array::from_vec(
        features
            .iter()
            .map(|feature| &feature.token_ids)
            .map(|input| {
                let mut attention_mask = Vec::with_capacity(max_len);
                attention_mask.resize(input.len(), 1);
                attention_mask.resize(max_len, 0);
                attention_mask
            })
            .collect(),
    );

    let outputs = session.run::<_, f32>(&[&input_ids, &attention_masks])?;

    // // https://github.com/kerryeon/huggingface-onnx-tutorial/blob/master/rust/src/main.rs ref
    // let start_logits = outputs[0].to_shape(input_shape)?;
    // let end_logits = outputs[1].to_shape(input_shape)?;
    // let answer = find_answer(&input_ids, &start_logits, &end_logits);
    // dbg!(answer
    //     .iter()
    //     .map(|row| tokenizer
    //         .decode(row.as_slice().unwrap(), true, true)
    //         .trim()
    //         .to_string())
    //     .collect::<Vec<_>>())

    // let results = session.run(token_outputs).unwrap();
    // for token in results {
    //     println!("{token:?}");