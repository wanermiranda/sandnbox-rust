mod models;
mod tokens;
mod utilities;
use colored::Colorize;

use models::{
    xlm_roberta_onnx::{self, predict_sentiment},
    xlm_roberta_rustbert,
};
use terminal_menu::mut_menu;
use utilities::tokens::{bench_tonizers, generate_random_tokens};

use tokens::bert_rustbert;

fn main() {
    use terminal_menu::{button, label, menu, run};
    let menu = menu(vec![
        label("--------------"),
        label("There are some basic tasks on this sandbox."),
        label("---------------"),
        button("ner_models"),
        button("others"),
        button("exit"),
    ]);
    run(&menu);
    let mm = mut_menu(&menu);

    match mm.selected_item_name() {
        "onnx" => ner_models(),
        "others" => other_models(),
        i => println!("Menu item {i} not found."),
    }
}

fn ner_models() {
    let input = [
        "My name is Amélie. I live in Москва.",
        "Chongqing is a city in China.",
        "Meu nome é Waner e moro no Brasil.",
        "My name is Mario and I live in Canada.",
    ];

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
        println!("{}", "NER Using Roberta rust_bert".bold().blue());

        let token_classification_model = xlm_roberta_rustbert::build_model().unwrap();

        timeit!(token_classification_model.predict(&input));

        let token_outputs = token_classification_model.predict(&input);

        for token in token_outputs {
            println!("{token:?}");
        }
    }

    {
        println!("{}", "NER Using Roberta onnx".bold().blue());

        let token_classification_model = xlm_roberta_onnx::build_model();

        timeit!(xlm_roberta_onnx::predict(
            &input,
            &token_classification_model
        ));

        let token_outputs = xlm_roberta_onnx::predict(&input, &token_classification_model);

        let parsed_ouputs = xlm_roberta_onnx::parse_tokens(&token_outputs);
        for token in parsed_ouputs {
            println!("{token:?}");
        }
    }
}

fn other_models() {
    {
        println!("{}", "Generating Tokens Dry Run ".bold().blue());
        let temp_token_file = "_tok.json";
        let num_tokens = 10_000;
        timeit!(generate_random_tokens(temp_token_file, num_tokens).unwrap());
        timeit!(bench_tonizers(temp_token_file).unwrap());
        std::fs::remove_file(temp_token_file).unwrap();
    }

    {
        let texts = vec!["You are awesome", "You are bad"];

        timeit!(predict_sentiment(&texts));

        let responses = predict_sentiment(&texts);
        let res_positive = responses.get(0).unwrap();
        let res_negative = responses.get(1).unwrap();

        assert!(res_positive[0] < res_positive[1]);
        println!("{} {:?}", texts[0], res_positive);
        assert!(res_negative[0] > res_negative[1]);
        println!("{} {:?}", texts[1], res_negative);
    }
}
