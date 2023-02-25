mod utilities;
use colored::Colorize;

mod models;

mod tokens;

use sandbox_rust::models::xlm_roberta_rustbert;
use terminal_menu::mut_menu;
use utilities::tokens::{bench_tonizers, generate_random_tokens};

use crate::tokens::bert_rustbert;

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
        "onnx" => onnx_models(),
        "rust_bert" => rust_bert_models(),
        i => println!("Menu item {i} not found."),
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

        let token_classification_model = bert_rustbert::build_model().unwrap();

        timeit!(token_classification_model.predict(&input));

        let token_outputs = token_classification_model.predict(&input);
        for token in token_outputs {
            println!("{token:?}");
        }
    }

    {
        println!("{}", "NER Using Roberta".bold().blue());

        let token_classification_model = xlm_roberta_rustbert::build_model().unwrap();

        timeit!(token_classification_model.predict(&input));

        let token_outputs = token_classification_model.predict(&input);

        for token in token_outputs {
            println!("{token:?}");
        }
    }
}

fn onnx_models() {
    // let text_positive = "You are awesome".to_string();

    // let res_positive = bert_onnx::predict_ner(&text_positive, true);
    // println!("{} {:?}", text_positive, res_positive);

    // let text_negative = "You are bad".to_string();

    // let res_negative = bert_onnx::predict_single(&text_negative, true);
    // println!("{} {:?}", text_negative, res_negative);
}
