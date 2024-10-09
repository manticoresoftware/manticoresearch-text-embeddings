use std::ffi::c_void;
use std::os::raw::c_char;
use crate::model::Model;
use crate::ffi::FloatVec;

#[repr(transparent)]
pub struct TextModel(*const c_void);

impl TextModel {
	pub extern "C" fn load_model(name_ptr: *const c_char, name_len: usize) -> Self {
		let name = unsafe {
			let slice = std::slice::from_raw_parts(name_ptr as *mut u8, name_len);
			std::str::from_utf8_unchecked(slice)
		};

		let model = Model::create(name.to_string(), None, None);
		TextModel(Box::into_raw(Box::new(model)) as *const c_void)
	}

	pub extern "C" fn delete_model(self) {
		unsafe {
			drop(Box::from_raw(self.0 as *mut Model));
		}
	}

	fn as_model(&self) -> &Model {
		unsafe { & *(self.0 as *const Model) }
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
