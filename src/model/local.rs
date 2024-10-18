use super::TextModel;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::{DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper, Tokenizer, TokenizerImpl};
use std::path::PathBuf;

use crate::utils::{get_max_input_length, get_hidden_size, normalize, chunk_input_tokens, get_mean_vector};

struct ModelInfo {
	config_path: PathBuf,
	tokenizer_path: PathBuf,
	weights_path: PathBuf,
	use_pth: bool,
}

fn build_model_info(
	cache_path: PathBuf,
	model_id: &str,
	revision: &str,
	use_pth: bool,
) -> Result<ModelInfo> {
	let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string());
	let api = ApiBuilder::new().with_cache_dir(cache_path).build()?;
	let api = api.repo(repo);

	let config_path = api.get("config.json")?;
	let tokenizer_path = api.get("tokenizer.json")?;
	let weights_path = if use_pth {
		api.get("pytorch_model.bin")?
	} else {
		api.get("model.safetensors")?
	};
	Ok(ModelInfo{
		config_path,
		tokenizer_path,
		weights_path,
		use_pth,
	})
}

fn build_model_and_tokenizer(
	model: ModelInfo,
	device: Device,
) -> Result<(BertModel, Tokenizer, usize, usize)> {
	let config = std::fs::read_to_string(model.config_path)?;
	let max_input_len = get_max_input_length(&config)?;
	let hidden_size = get_hidden_size(&config)?;

	let mut config: Config = serde_json::from_str(&config)?;
	let tokenizer: Tokenizer = Tokenizer::from_file(model.tokenizer_path).map_err(E::msg)?;

	let vb = if model.use_pth {
		VarBuilder::from_pth(&model.weights_path, DTYPE, &device)?
	} else {
		unsafe { VarBuilder::from_mmaped_safetensors(&[model.weights_path], DTYPE, &device)? }
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
	pub fn new(model_id: &str, cache_path: PathBuf, use_gpu: bool) -> Self {
		let revision = "main";
		let use_pth = false;
		let device = if use_gpu {
			Device::new_cuda(0).unwrap()
		} else {
			Device::Cpu
		};
		let result = std::panic::catch_unwind(|| {
			let model_info = build_model_info(cache_path, model_id, revision, use_pth).unwrap();
			let (model, mut tokenizer, max_input_len, hidden_size) =
			build_model_and_tokenizer(model_info, device).unwrap();
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
		});

		match result {
			Ok(model) => model,
			Err(e) => {
				let error = match e.downcast::<String>() {
					Ok(e) => *e,
					Err(_) => "Unknown error".to_string(),
				};
				panic!("Failed to create model: {}", error)
			},
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
		let cache_path = PathBuf::from(".cache/manticore");
		let local_model = LocalModel::new(model_id, cache_path, false);

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
		let cache_path = PathBuf::from(".cache/manticore");
		let local_model = LocalModel::new(model_id, cache_path, false);

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
		let cache_path = PathBuf::from(".cache/manticore");
		let local_model = LocalModel::new(model_id, cache_path, false);
		assert_eq!(local_model.get_hidden_size(), 384);
	}

	#[test]
	fn test_max_input_len() {
		let model_id = "sentence-transformers/all-MiniLM-L6-v2";
		let cache_path = PathBuf::from(".cache/manticore");
		let local_model = LocalModel::new(model_id, cache_path, false);
		assert_eq!(local_model.get_max_input_len(), 512);
	}
}
