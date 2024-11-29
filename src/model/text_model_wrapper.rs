use std::{ffi::c_void, ptr};
use std::os::raw::c_char;
use crate::model::{Model, TextModel, create_model, ModelOptions};

/// cbindgen:field-names=[m_pModel, m_szError]
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
pub struct FloatVecList {
	pub ptr: *const FloatVecResult,
	pub len: usize,
	pub cap: usize,
}

/// cbindgen:field-names=[m_tEmbedding, m_szError]
#[repr(C)]
pub struct FloatVecResult {
	vector: FloatVec,
	error: *mut c_char,
}

#[repr(C)]
pub struct StringItem {
    ptr: *const c_char,
    len: usize,
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

		match create_model(options) {
			Ok(model) => TextModelResult {
				model: Box::into_raw(Box::new(model)) as *mut c_void,
				error: ptr::null_mut(),
			},
			Err(e) => {
				let c_error = std::ffi::CString::new(e.to_string()).unwrap();
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

	pub extern "C" fn make_vect_embeddings(
		&self,
		texts: *const StringItem,
		count: usize
	) -> FloatVecList {
		let string_slice = unsafe {
			std::slice::from_raw_parts(texts, count)
		};

		let strings: Vec<&str> = string_slice.iter()
			.map(|item| unsafe {
				std::str::from_utf8_unchecked(
					std::slice::from_raw_parts(item.ptr as *const u8, item.len)
				)
			})
			.collect();

		let mut float_result_list: Vec<FloatVecResult> = Vec::new();
		let model = self.as_model();
		for text in strings.iter() {
			let embeddings = model.predict(text);
			let float_result = match embeddings {
				Ok(embeddings) => {
					let ptr = embeddings.as_ptr();
					let len = embeddings.len();
					let cap = embeddings.capacity();
					std::mem::forget(embeddings);
					let vec = FloatVec { ptr, len, cap };

					FloatVecResult { vector: vec, error: ptr::null_mut() }
				},
				Err(e) => {
					let c_error = std::ffi::CString::new(e.to_string()).unwrap();
					let vec = FloatVec { ptr: ptr::null(), len: 0, cap: 0 };
					FloatVecResult { vector: vec, error: c_error.into_raw() }
				}
			};
			float_result_list.push(float_result);
		}
		let vec_list = FloatVecList {
			ptr: float_result_list.as_ptr(),
			len: float_result_list.len(),
			cap: float_result_list.capacity()
		};
		std::mem::forget(float_result_list);
		vec_list
	}

	pub extern "C" fn free_vec_list(vec_list: FloatVecList) {
		unsafe {
			let slice = std::slice::from_raw_parts(vec_list.ptr, vec_list.len);

			for result in slice {
				// Free the FloatVec's inner buffer
				if !result.vector.ptr.is_null() {
					let _ = Vec::from_raw_parts(
						result.vector.ptr as *mut f32,
						result.vector.len,
						result.vector.cap
					);
				}

				// Free the error string if it exists
				if !result.error.is_null() {
					let _ = std::ffi::CString::from_raw(result.error);
				}
			}

			// Free the FloatVecList's array of FloatVecResult
			let _ = Vec::from_raw_parts(
				vec_list.ptr as *mut FloatVecResult,
				vec_list.len,
				vec_list.cap
			);
		}
	}

	pub extern "C" fn get_hidden_size(&self) -> usize {
		self.as_model().get_hidden_size()
	}

	pub extern "C" fn get_max_input_len(&self) -> usize {
		self.as_model().get_max_input_len()
	}
}

