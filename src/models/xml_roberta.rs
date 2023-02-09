extern crate rust_bert;

use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::ner::NERModel;
use rust_bert::pipelines::token_classification::TokenClassificationConfig;
use rust_bert::resources::RemoteResource;
use rust_bert::roberta::{RobertaConfigResources, RobertaModelResources, RobertaVocabResources};
use rust_bert::RustBertError;

// /// `NERModel::new(config)` creates a new NER model from the XML Roberta configuration `config`
/// It creates a `TokenClassificationConfig` object, which is then used to create a `NERModel` object
///
/// Returns:
///
/// A NERModel
pub fn build_model() -> Result<NERModel, RustBertError> {
    let config: TokenClassificationConfig = TokenClassificationConfig {
        model_type: ModelType::XLMRoberta,
        model_resource: Box::new(RemoteResource::from_pretrained(
            RobertaModelResources::XLM_ROBERTA_NER_EN,
        )),
        config_resource: Box::new(RemoteResource::from_pretrained(
            RobertaConfigResources::XLM_ROBERTA_NER_EN,
        )),
        vocab_resource: Box::new(RemoteResource::from_pretrained(
            RobertaVocabResources::XLM_ROBERTA_NER_EN,
        )),
        lower_case: false,
        // device: Device::cuda_if_available(),
        ..Default::default()
    };

    //    Create the model
    let token_classification_model = NERModel::new(config)?;
    Ok(token_classification_model)
}
