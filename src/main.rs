mod utilities;
use colored::Colorize;
use scanf::scanf; // needed for time

use models::xml_roberta;
use onnxruntime::{environment, LoggingLevel};
use terminal_menu::mut_menu;
use utilities::tokens::{bench_tonizers, generate_random_tokens};

mod models;
use models::token_bert;

fn main() {
    use terminal_menu::{button, label, menu, run};
    let menu = menu(vec![
        label("--------------"),
        label("There are some basic tasks on this sandbox."),
        label("---------------"),
        button("rust_bert"),
        button("onnx"),
        button("exit"),
    ]);
    run(&menu);
    let mm = mut_menu(&menu);

    match mm.selected_item_name() {
        "rust_bert" => rust_bert_models(),
        "onnx" => onnx_models(),
        i => println!("Menu item {} not found.", i),
    }
}

fn rust_bert_models() {
    let input = [
        "My name is Amélie. I live in Москва.",
        "Chongqing is a city in China.",
        "Meu nome é Waner e moro no Brasil.",
        "My name is Mario and I live in Canada.",
    ];

    {
        println!("{}", "Generating Tokens Dry Run ".bold().blue());
        let temp_token_file = "_tok.json";
        let num_tokens = 10_000;
        timeit!(generate_random_tokens(temp_token_file, num_tokens).unwrap());
        timeit!(bench_tonizers(temp_token_file).unwrap());
        std::fs::remove_file(temp_token_file).unwrap();
    }

    {
        println!("{}", "NER Using BERT".bold().blue());

        let token_classification_model = token_bert::build_model().unwrap();

        timeit!(token_classification_model.predict(&input));

        let token_outputs = token_classification_model.predict(&input);
        for token in token_outputs {
            println!("{token:?}");
        }
    }

    {
        println!("{}", "NER Using Roberta".bold().blue());

        let token_classification_model = xml_roberta::build_model().unwrap();

        timeit!(token_classification_model.predict(&input));

        let token_outputs = token_classification_model.predict(&input);

        for token in token_outputs {
            println!("{token:?}");
        }
    }
}

fn onnx_models() {
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

    println!("Inputs:");
    for (index, input) in session.inputs.iter().enumerate() {
        println!(
            "  {}:\n    name = {}\n    type = {:?}\n    dimensions = {:?}",
            index, input.name, input.input_type, input.dimensions
        )
    }

    println!("Outputs:");
    for (index, output) in session.outputs.iter().enumerate() {
        println!(
            "  {}:\n    name = {}\n    type = {:?}\n    dimensions = {:?}",
            index, output.name, output.output_type, output.dimensions
        );
    }
}
