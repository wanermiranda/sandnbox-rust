mod utilities;
use colored::Colorize;

mod models;
use models::token_bert;
use models::xml_roberta;

mod tokens;

use terminal_menu::mut_menu;
use utilities::tokens::{bench_tonizers, generate_random_tokens};

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
    // https://colab.research.google.com/github/neuml/txtai/blob/master/examples/18_Export_and_run_models_with_ONNX.ipynb#scrollTo=_8fdRvO1fFBm
    //   println!(
    //       "outputs: {:#?}",
    //       outputs
    //           .pop()
    //           .unwrap()
    //           .map_axis(ndarray::Axis(1), |x| x[0] > x[1])
    //           .map(|x| match x {
    //               True => "Open",
    //               False => "Not Open",
    //           })
    //   );
    //   println!("outputs: {:#?}\n", &outputs);
    // find and display the max value with its index
}
