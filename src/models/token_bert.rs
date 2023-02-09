extern crate rust_bert;

use rust_bert::bert::{BertConfigResources, BertModelResources, BertVocabResources};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::ner::NERModel;
use rust_bert::pipelines::token_classification::{
    LabelAggregationOption, TokenClassificationConfig,
};
use rust_bert::resources::RemoteResource;
use rust_bert::RustBertError;

// /// `NERModel::new(config)` creates a new NER model from the BertModel configuration `config`
/// It creates a `TokenClassificationConfig` object, which is then used to create a `NERModel` object
///
/// Returns:
///
/// A NERModel
pub fn build_model() -> Result<NERModel, RustBertError> {
    let config = TokenClassificationConfig::new(
        ModelType::Bert,
        RemoteResource::from_pretrained(BertModelResources::BERT_NER),
        RemoteResource::from_pretrained(BertConfigResources::BERT_NER),
        RemoteResource::from_pretrained(BertVocabResources::BERT_NER),
        None,  //merges resource only relevant with ModelType::Roberta
        false, //lowercase
        false,
        None,
        LabelAggregationOption::Mode,
    );

    //    Create the model
    let token_classification_model = NERModel::new(config)?;
    Ok(token_classification_model)
}
