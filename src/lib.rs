use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};

use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde_json::Value;
use tokenizers::{DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper, Tokenizer, TokenizerImpl};
use std::{ffi::c_void, os::raw::c_char};
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

/// Get maximum input length for sequence for the current model
fn get_max_input_length(contents: &str) -> Result<usize> {
	let config: Value = serde_json::from_str(&contents)?;
	let max_length = config["max_position_embeddings"]
		.as_u64()
		.ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Max position embeddings not found"))?;
	Ok(max_length as usize)
}

fn get_hidden_size(contents: &str) -> Result<usize> {
	let config: Value = serde_json::from_str(&contents)?;
	let max_length = config["hidden_size"]
		.as_u64()
		.ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Hidden size not found"))?;
	Ok(max_length as usize)
}

pub fn normalize(v: &mut Vec<f32>) {
	let length: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
	v.iter_mut().for_each(|x| *x /= length);
}

fn chunk_input_tokens(tokens: &[u32], max_seq_len: usize, stride: usize) -> Vec<Vec<u32>> {
	if tokens.len() <= max_seq_len {
		return vec![tokens.to_vec()];
	}

	let mut chunks = Vec::new();
	let mut start = 0;
	let len = tokens.len();

	while start < len {
		let end = std::cmp::min(start + max_seq_len, len);
		chunks.push(tokens[start..end].to_vec());
		start += std::cmp::min(max_seq_len - stride, len - start);
	}

	chunks
}

fn get_mean_vector(results: &Vec<Vec<f32>>) -> Vec<f32> {
	if results.is_empty() {
		return Vec::new();
	}

	let num_cols = results[0].len();
	let mut mean_vector = vec![0.0; num_cols];

	let mut weight_sum = 0.0;

	for (i, row) in results.iter().enumerate() {
		let weight = if i == 0 { 1.2 } else { 1.0 }; // Adjust the weight for the first chunk here
		weight_sum += weight;

		for (j, val) in row.iter().enumerate() {
			mean_vector[j] += weight * val;
		}
	}

	for val in &mut mean_vector {
		*val /= weight_sum;
	}

	mean_vector
}

pub struct Model {
	model: BertModel,
	tokenizer: TokenizerImpl<ModelWrapper, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper>,
	max_input_len: usize,
	hidden_size: usize,
}

impl Model {
	pub fn create(model_id: String, revision: Option<String>, use_pth: Option<bool>) -> Self {
		let revision = revision.unwrap_or("main".to_string());
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
		for chunk in &chunks {
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
}

#[repr(transparent)]
struct TextModel(*const c_void);

#[repr(C)]
struct FloatVec {
    ptr: *mut f32,
    len: usize,
	cap: usize,
}


impl TextModel {

	extern "C" fn load_model(name_ptr: *const c_char, name_len: usize) -> Self {
		let name = unsafe {
			let slice = std::slice::from_raw_parts(name_ptr as *mut u8, name_len);
			std::str::from_utf8_unchecked(slice)
		};

		let model = Model::create(name.to_string(), None, None);
		TextModel(Box::into_raw(Box::new(model)) as *const c_void)
	}

	extern "C" fn delete_model(self) {
		unsafe {
			drop(Box::from_raw(self.0 as *mut Model));
		}
	}

	fn as_model(&self) -> &Model {
		unsafe { & *(self.0 as *const Model) }
	}

	fn make_embeddings(self, text_ptr: *const c_char, text_len: usize) -> Vec<f32> {
		let text = unsafe {
			let slice = std::slice::from_raw_parts(text_ptr as *mut u8, text_len);
			std::str::from_utf8_unchecked(slice)
		};

		self.as_model().predict(text)
	}

	extern "C" fn make_vect_embeddings(self, text_ptr: *const c_char, text_len: usize) -> FloatVec {
		let mut embeddings = self.make_embeddings(text_ptr,text_len);//.into_boxed_slice();
		let ptr = embeddings.as_mut_ptr();
		let len = embeddings.len();
		let cap = embeddings.capacity();
		std::mem::forget(embeddings);
		FloatVec { ptr, len, cap }
	}

	extern "C" fn delete_vec(buf: FloatVec) {
//		let s = unsafe { std::slice::from_raw_parts_mut(buf.ptr, buf.len) };
//		unsafe {
//			drop(Box::from_raw(s as *mut [f32]));
//		}
		unsafe { Vec::from_raw_parts(buf.ptr, buf.len, buf.cap) };
	}

	extern "C" fn get_hidden_size(self) -> usize {
		self.as_model().get_hidden_size()
	}


	extern "C" fn get_max_input_len(self) -> usize {
		self.as_model().get_max_input_len()
	}
}


type LoadModelFn = extern "C" fn (*const c_char, usize) -> TextModel;
type DeleteModelFn = extern "C" fn (TextModel);
type MakeVectEmbeddingsFn = extern "C" fn (TextModel, *const c_char, usize) -> FloatVec;
type DeleteVecFn = extern "C" fn (FloatVec);
type GetLenFn = extern "C" fn (TextModel) -> usize;

#[allow(unused)]
#[no_mangle]
#[repr(C)]
pub struct EmbeddLib {
    size: usize,
    load_model: LoadModelFn,
    delete_model: DeleteModelFn,
    make_vect_embeddings: MakeVectEmbeddingsFn,
    delete_vec: DeleteVecFn,
	get_hidden_size: GetLenFn,
	get_max_input_size: GetLenFn,
}

impl Default for EmbeddLib {
    fn default() -> Self {
        Self {
            size: std::mem::size_of::<Self>(),
            load_model: TextModel::load_model,
            delete_model: TextModel::delete_model,
            make_vect_embeddings: TextModel::make_vect_embeddings,
            delete_vec: TextModel::delete_vec,
			get_hidden_size: TextModel::get_hidden_size,
			get_max_input_size: TextModel::get_max_input_len,
        }
    }
}

#[no_mangle]
pub extern "C" fn GetLibFuncs() -> EmbeddLib {
    EmbeddLib::default()
}
