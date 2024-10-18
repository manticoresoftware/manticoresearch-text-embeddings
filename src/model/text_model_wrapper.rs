use std::{ffi::c_void, ptr};
use std::os::raw::c_char;
use crate::model::{Model, TextModel, create_model, ModelOptions};

#[repr(C)]
pub struct TextModelResult {
	model: *mut c_void,
	error: *mut c_char,
}

#[repr(transparent)]
pub struct TextModelWrapper(*mut c_void);


#[repr(C)]
pub struct FloatVec {
	pub ptr: *const f32,
	pub len: usize,
	pub cap: usize,
}

#[repr(C)]
pub struct FloatVecResult {
	vector: FloatVec,
	error: *mut c_char,
}

impl TextModelWrapper {
	pub extern "C" fn load_model(
		name_ptr: *const c_char,
		name_len: usize,
		cache_path_ptr: *const c_char,
		cache_path_len: usize,
		api_key_ptr: *const c_char,
		api_key_len: usize,
		use_gpu: bool,
	) -> TextModelResult {
		let result = std::panic::catch_unwind(|| {
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
				use_gpu: Some(use_gpu),
			};

			create_model(options)
		});

		match result {
			Ok(model) => TextModelResult {
				model: Box::into_raw(Box::new(model)) as *mut c_void,
				error: ptr::null_mut(),
			},
			Err(e) => {
				let error = match e.downcast::<String>() {
					Ok(e) => *e,
					Err(_) => "Unknown error".to_string(),
				};
				let c_error = std::ffi::CString::new(error).unwrap();
				TextModelResult {
					model: ptr::null_mut(),
					error: c_error.into_raw(),
				}
			}
		}
	}

	pub extern "C" fn free_model_result(res: TextModelResult) {
		unsafe {
			if !res.model.is_null() {
				drop(Box::from_raw(res.model as *mut Model));
			}

			if !res.error.is_null() {
				let _ = std::ffi::CString::from_raw(res.error);
			}
		}
	}

	fn as_model(&self) -> &Model {
		unsafe { &*(self.0 as *const Model) }
	}

	pub extern "C" fn make_vect_embeddings(&self, text_ptr: *const c_char, text_len: usize) -> FloatVecResult {
		let result = std::panic::catch_unwind(|| {
			let text = unsafe {
				std::str::from_utf8_unchecked(std::slice::from_raw_parts(text_ptr as *const u8, text_len))
			};

			let embeddings = self.as_model().predict(text);
			let ptr = embeddings.as_ptr();
			let len = embeddings.len();
			let cap = embeddings.capacity();
			std::mem::forget(embeddings);
			FloatVec { ptr, len, cap }
		});
		match result {
			Ok(embeddings) => FloatVecResult {
				vector: embeddings,
				error: ptr::null_mut(),
			},
			Err(e) => {
				let error = match e.downcast::<String>() {
					Ok(e) => *e,
					Err(_) => "Unknown error".to_string(),
				};
				let c_error = std::ffi::CString::new(error).unwrap();
				FloatVecResult {
					vector: FloatVec { ptr: ptr::null(), len: 0, cap: 0 },
					error: c_error.into_raw(),
				}
			}
		}
	}

	pub extern "C" fn free_vec_result(vec: FloatVecResult) {
		unsafe {
			Vec::from_raw_parts(vec.vector.ptr as *mut FloatVec, vec.vector.len, vec.vector.cap);

			if !vec.error.is_null() {
				let _ = std::ffi::CString::from_raw(vec.error);
			}
		}
	}

	pub extern "C" fn get_hidden_size(&self) -> usize {
		self.as_model().get_hidden_size()
	}

	pub extern "C" fn get_max_input_len(&self) -> usize {
		self.as_model().get_max_input_len()
	}
}

