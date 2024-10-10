use super::TextModel;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper, Tokenizer, TokenizerImpl};

use crate::utils::{get_max_input_length, get_hidden_size, normalize, chunk_input_tokens, get_mean_vector};

fn build_model_and_tokenizer(
	model_id: &str,
	revision: &str,
	use_pth: bool,
) -> Result<(BertModel, Tokenizer, usize, usize)> {
	let device = Device::Cpu;
	let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string());
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

pub struct LocalModel {
	model: BertModel,
	tokenizer: Tokenizer,
	max_input_len: usize,
	hidden_size: usize,
}

impl LocalModel {
	pub fn new(model_id: &str) -> Self {
		let revision = "main";
		let use_pth = false;
		let (model, mut tokenizer, max_input_len, hidden_size) =
		build_model_and_tokenizer(model_id, revision, use_pth).unwrap();
		let tokenizer = tokenizer
			.with_padding(None)
			.with_truncation(None)
			.map_err(E::msg)
			.unwrap();

		Self {
			model,
			tokenizer: tokenizer.clone().into(),
			max_input_len,
			hidden_size,
		}
	}
}

impl serde::Serialize for LocalModel {
	fn serialize<S>(&self, serializer: S) -> std::prelude::v1::Result<S::Ok, S::Error> where
		S: serde::Serializer, {
		<TokenizerImpl<ModelWrapper, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper> as serde::Serialize>::serialize(&self.tokenizer, serializer)
	}
}

impl TextModel for LocalModel {
	fn predict(&self, text: &str) -> Vec<f32> {
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

	fn get_hidden_size(&self) -> usize {
		self.hidden_size
	}

	fn get_max_input_len(&self) -> usize {
		self.max_input_len
	}
}


#[cfg(test)]
mod tests {
	use super::*;
	use approx::assert_abs_diff_eq;

	fn check_embedding_properties(embedding: &[f32], expected_len: usize) {
		assert_eq!(embedding.len(), expected_len);

		// Check if the embedding is normalized (L2 norm should be close to 1)
		let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
		assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-6);
	}

	#[test]
	fn test_all_minilm_l6_v2() {
		let model_id = "sentence-transformers/all-MiniLM-L6-v2";
		let local_model = LocalModel::new(model_id);

		let test_sentences = [
			"This is a test sentence.",
			"Another sentence to encode.",
			"Sentence transformers are awesome!",
		];

		for sentence in &test_sentences {
			let embedding = local_model.predict(sentence);
			check_embedding_properties(&embedding, local_model.get_hidden_size());
		}
	}

	#[test]
	fn test_embedding_consistency() {
		let model_id = "sentence-transformers/all-MiniLM-L6-v2";
		let local_model = LocalModel::new(model_id);

		let sentence = "This is a test sentence.";
		let embedding1 = local_model.predict(sentence);
		let embedding2 = local_model.predict(sentence);

		for (e1, e2) in embedding1.iter().zip(embedding2.iter()) {
			assert_abs_diff_eq!(e1, e2, epsilon = 1e-6);
		}
	}

	#[test]
	fn test_hidden_size() {
		let model_id = "sentence-transformers/all-MiniLM-L6-v2";
		let local_model = LocalModel::new(model_id);
		assert_eq!(local_model.get_hidden_size(), 384);
	}

	#[test]
	fn test_max_input_len() {
		let model_id = "sentence-transformers/all-MiniLM-L6-v2";
		let local_model = LocalModel::new(model_id);
		assert_eq!(local_model.get_max_input_len(), 512);
	}
}
