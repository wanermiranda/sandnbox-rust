mod utilities;

use colored::Colorize; // needed for time

use models::xml_roberta;
use utilities::tokens::{bench_tonizers, generate_random_tokens};

mod models;
use models::token_bert;

fn main() {
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
