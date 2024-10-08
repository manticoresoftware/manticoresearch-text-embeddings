use std::os::raw::c_char;
use crate::text_model::TextModel;

#[repr(C)]
pub struct FloatVec {
	pub ptr: *const f32,
	pub len: usize,
	pub cap: usize,
}

type LoadModelFn = extern "C" fn (*const c_char, usize) -> TextModel;
type DeleteModelFn = extern "C" fn (TextModel);
type MakeVectEmbeddingsFn = extern "C" fn (&TextModel, *const c_char, usize) -> FloatVec;
type DeleteVecFn = extern "C" fn (FloatVec);
type GetLenFn = extern "C" fn (&TextModel) -> usize;

#[allow(unused)]
#[no_mangle]
#[repr(C)]
pub struct EmbedLib {
	size: usize,
	load_model: LoadModelFn,
	delete_model: DeleteModelFn,
	make_vect_embeddings: MakeVectEmbeddingsFn,
	delete_vec: DeleteVecFn,
	get_hidden_size: GetLenFn,
	get_max_input_size: GetLenFn,
}

const LIB: EmbedLib = EmbedLib {
	size: std::mem::size_of::<EmbedLib>(),
	load_model: TextModel::load_model,
	delete_model: TextModel::delete_model,
	make_vect_embeddings: TextModel::make_vect_embeddings,
	delete_vec: TextModel::delete_vec,
	get_hidden_size: TextModel::get_hidden_size,
	get_max_input_size: TextModel::get_max_input_len,
};

#[no_mangle]
pub extern "C" fn GetLibFuncs() -> *const EmbedLib {
	&LIB
}
