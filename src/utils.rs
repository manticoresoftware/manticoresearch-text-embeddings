use serde_json::Value;
use anyhow::Result;

/// Get maximum input length for sequence for the current model
pub fn get_max_input_length(contents: &str) -> Result<usize> {
	let config: Value = serde_json::from_str(&contents)?;
	let max_length = config["max_position_embeddings"]
		.as_u64()
		.ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Max position embeddings not found"))?;
	Ok(max_length as usize)
}

pub fn get_hidden_size(contents: &str) -> Result<usize> {
	let config: Value = serde_json::from_str(&contents)?;
	let max_length = config["hidden_size"]
		.as_u64()
		.ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Hidden size not found"))?;
	Ok(max_length as usize)
}

#[inline]
pub fn normalize(v: &mut Vec<f32>) {
	let length: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
	v.iter_mut().for_each(|x| *x /= length);
}

pub fn chunk_input_tokens(tokens: &[u32], max_seq_len: usize, stride: usize) -> Vec<Vec<u32>> {
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

pub fn get_mean_vector(results: &Vec<Vec<f32>>) -> Vec<f32> {
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
