use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper, Tokenizer, TokenizerImpl};

use crate::utils::{get_max_input_length, get_hidden_size, normalize, chunk_input_tokens, get_mean_vector};

fn build_model_and_tokenizer(
	model_id: String,
	revision: String,
	use_pth: bool,
) -> Result<(BertModel, Tokenizer, usize, usize)> {
	let device = Device::Cpu;
	let repo = Repo::with_revision(model_id, RepoType::Model, revision);
	let (config_filename, tokenizer_filename, weights_filename) = {
		let api = Api::new()?;
		let api = api.repo(repo);
		let config = api.get("config.json")?;
		let tokenizer = api.get("tokenizer.json")?;
		let weights = if use_pth {
			api.get("pytorch_model.bin")?
		} else {
			api.get("model.safetensors")?
		};
		(config, tokenizer, weights)
	};
	let config = std::fs::read_to_string(config_filename)?;
	let max_input_len = get_max_input_length(&config)?;
	let hidden_size = get_hidden_size(&config)?;
	let mut config: Config = serde_json::from_str(&config)?;
	let tokenizer: Tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

	let vb = if use_pth {
		VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
	} else {
		unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
	};
	config.hidden_act = HiddenAct::GeluApproximate;
	let model = BertModel::load(vb, &config)?;
	Ok((model, tokenizer, max_input_len, hidden_size))
}

pub struct Model {
	model: BertModel,
	tokenizer: TokenizerImpl<ModelWrapper, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper>,
	max_input_len: usize,
	hidden_size: usize,
}

impl serde::Serialize for Model {
	fn serialize<S>(&self, serializer: S) -> std::prelude::v1::Result<S::Ok, S::Error> where
		S: serde::Serializer, {
		<TokenizerImpl<ModelWrapper, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper> as serde::Serialize>::serialize(&self.tokenizer, serializer)
	}
}

impl Model {
	pub fn create(model_id: String, revision: Option<String>, use_pth: Option<bool>) -> Self {
		let revision = revision.unwrap_or(String::from("main"));
		let use_pth = use_pth.unwrap_or(false);
		let (model, mut tokenizer, max_input_len, hidden_size) =
		build_model_and_tokenizer(model_id, revision, use_pth).unwrap();
		let tokenizer = tokenizer
			.with_padding(None)
			.with_truncation(None)
			.map_err(E::msg)
			.unwrap();

		Model {
			model,
			tokenizer: tokenizer.clone(),
			max_input_len,
			hidden_size,
		}
	}

	pub fn get_max_input_len(&self) -> usize {
		self.max_input_len
	}

	pub fn get_hidden_size(&self) -> usize {
		self.hidden_size
	}

	pub fn predict(&self, text: &str) -> Vec<f32> {
		let device = &self.model.device;
		let tokens = self.tokenizer
			.encode(text, true)
			.map_err(E::msg)
			.unwrap()
			.get_ids()
			.to_vec();
		let chunks = chunk_input_tokens(&tokens, self.max_input_len, (self.max_input_len / 10) as usize);
		let mut results: Vec<Vec<f32>> = Vec::new();
		for chunk in chunks.iter() {
			let token_ids = Tensor::new(&chunk[..], device).unwrap().unsqueeze(0).unwrap();
			let token_type_ids = token_ids.zeros_like().unwrap();
			let embeddings = self.model.forward(&token_ids, &token_type_ids).unwrap();

			// Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
			let (n_sentences, n_tokens, _hidden_size) = embeddings.dims3().unwrap();
			let embeddings = (embeddings.sum(1).unwrap() / (n_tokens as f64)).unwrap();

			for j in 0..n_sentences {
				let e_j = embeddings.get(j).unwrap();
				let mut emb: Vec<f32> = e_j.to_vec1().unwrap();
				normalize(&mut emb);
				results.push(emb);
				break;
			}
		}
		get_mean_vector(&results)
	}

	pub fn tokenizer(&self) -> &TokenizerImpl<ModelWrapper, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper> {
		&self.tokenizer
	}
}
