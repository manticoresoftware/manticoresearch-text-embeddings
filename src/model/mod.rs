mod openai;
mod local;
pub mod text_model_wrapper;

pub trait TextModel {
	fn predict(&self, text: &str) -> Vec<f32>;
	fn get_hidden_size(&self) -> usize;
	fn get_max_input_len(&self) -> usize;
}

#[repr(C)]
pub struct FloatVec {
	pub ptr: *const f32,
	pub len: usize,
	pub cap: usize,
}

#[repr(C)]
pub struct ModelOptions {
	model_id: String,
	api_key: Option<String>,
}

#[repr(C)]
pub enum Model {
	OpenAI(openai::OpenAIModel),
	Local(local::LocalModel),
}

impl TextModel for Model {
	fn predict(&self, text: &str) -> Vec<f32> {
		match self {
			Model::OpenAI(m) => m.predict(text),
			Model::Local(m) => m.predict(text),
		}
	}

	fn get_hidden_size(&self) -> usize {
		match self {
			Model::OpenAI(m) => m.get_hidden_size(),
			Model::Local(m) => m.get_hidden_size(),
		}
	}

	fn get_max_input_len(&self) -> usize {
		match self {
			Model::OpenAI(m) => m.get_max_input_len(),
			Model::Local(m) => m.get_max_input_len(),
		}
	}
}

pub fn create_model(options: ModelOptions) -> Model {
	let model_id = options.model_id.as_str();
	if model_id.starts_with("openai/") {
		Model::OpenAI(
			openai::OpenAIModel::new(
				model_id,
				options.api_key
					.unwrap_or(String::new())
					.as_str()
			)
		)
	} else {
		Model::Local(local::LocalModel::new(model_id))
	}
}

