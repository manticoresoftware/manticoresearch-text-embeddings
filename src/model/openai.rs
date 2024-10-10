use super::TextModel;
use reqwest::blocking::Client;

pub struct OpenAIModel {
	client: Client,
	model: String,
	api_key: String,
}

fn validate_model(model: &str) -> Result<(), String> {
	match model {
		"text-embedding-ada-002" | "text-embedding-3-small" | "text-embedding-3-large" => Ok(()),
		_ => Err(format!("Invalid model: {}", model)),
	}
}

fn validate_api_key(api_key: &str) -> Result<(), String> {
	if api_key.is_empty() {
		return Err("API key is required".to_string());
	}

	// now match that it starts with sk-
	if !api_key.starts_with("sk-") {
		return Err("API key must start with sk-".to_string());
	}

	Ok(())
}

impl OpenAIModel {
	pub fn new(model_id: &str, api_key: &str) -> Self {
		let model = model_id.trim_start_matches("openai/").to_string();
		validate_model(&model).expect("Invalid model");
		validate_api_key(&api_key).expect("Invalid API key");
		Self {
			client: Client::new(),
			model,
			api_key: api_key.to_string(),
		}
	}
}

impl TextModel for OpenAIModel {
	fn predict(&self, text: &str) -> Vec<f32> {
		let url = "https://api.openai.com/v1/embeddings";

		let request_body = serde_json::json!({
			"input": text,
			"model": self.model,
		});

		let response = self.client
			.post(url)
			.header("Authorization", format!("Bearer {}", self.api_key))
			.header("Content-Type", "application/json")
			.json(&request_body)
			.send()
			.expect("Failed to send request");

		let response_body: serde_json::Value = response
			.json()
			.expect("Failed to parse response");

		let embedding = response_body["data"][0]["embedding"]
			.as_array()
			.expect("Failed to get embedding array")
			.iter()
			.map(|v| v.as_f64().unwrap() as f32)
			.collect();

		embedding
	}

	fn get_hidden_size(&self) -> usize {
		match self.model.as_str() {
			"text-embedding-ada-002" => 768,
			"text-embedding-3-small" => 1536,
			"text-embedding-3-large" => 3072,
			_ => panic!("Unknown model"),
		}
	}

	fn get_max_input_len(&self) -> usize {
		8192
	}
}

