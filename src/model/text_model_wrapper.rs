use std::ffi::c_void;
use std::os::raw::c_char;
use crate::model::{Model, TextModel, create_model, ModelOptions, FloatVec};

#[repr(transparent)]
pub struct TextModelWrapper(*mut c_void);

impl TextModelWrapper {
	pub extern "C" fn load_model(
		name_ptr: *const c_char,
		name_len: usize,
		cache_path_ptr: *const c_char,
		cache_path_len: usize,
		api_key_ptr: *const c_char,
		api_key_len: usize,
	) -> Self {
		let name = unsafe {
			let slice = std::slice::from_raw_parts(name_ptr as *mut u8, name_len);
			std::str::from_utf8_unchecked(slice)
		};

		let cache_path = unsafe {
			let slice = std::slice::from_raw_parts(cache_path_ptr as *mut u8, cache_path_len);
			std::str::from_utf8_unchecked(slice)
		};

		let api_key = unsafe {
			let slice = std::slice::from_raw_parts(api_key_ptr as *mut u8, api_key_len);
			std::str::from_utf8_unchecked(slice)
		};

		let options = ModelOptions {
			model_id: name.to_string(),
			cache_path: if cache_path.is_empty() {
				None
			} else {
				Some(cache_path.to_string())
			},
			api_key: if api_key.is_empty() {
				None
			} else {
				Some(api_key.to_string())
			},
		};

		let model = create_model(options);
		TextModelWrapper(Box::into_raw(Box::new(model)) as *mut c_void)
	}

	pub extern "C" fn delete_model(self) {
		unsafe {
			drop(Box::from_raw(self.0 as *mut Model));
		}
	}

	fn as_model(&self) -> &Model {
		unsafe { &*(self.0 as *const Model) }
	}

	pub extern "C" fn make_vect_embeddings(&self, text_ptr: *const c_char, text_len: usize) -> FloatVec {
		let text = unsafe {
			std::str::from_utf8_unchecked(std::slice::from_raw_parts(text_ptr as *const u8, text_len))
		};

		let embeddings = self.as_model().predict(text);
		let ptr = embeddings.as_ptr();
		let len = embeddings.len();
		let cap = embeddings.capacity();
		std::mem::forget(embeddings);

		FloatVec { ptr, len, cap }
	}

	pub extern "C" fn delete_vec(vec: FloatVec) {
		unsafe {
			Vec::from_raw_parts(vec.ptr as *mut f32, vec.len, vec.cap);
		}
	}

	pub extern "C" fn get_hidden_size(&self) -> usize {
		self.as_model().get_hidden_size()
	}

	pub extern "C" fn get_max_input_len(&self) -> usize {
		self.as_model().get_max_input_len()
	}
}
