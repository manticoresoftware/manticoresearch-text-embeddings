use std::os::raw::c_char;
use crate::model::FloatVec;
use crate::model::text_model_wrapper::TextModelWrapper;

type LoadModelFn = extern "C" fn (*const c_char, usize, *const c_char, usize) -> TextModelWrapper;
type DeleteModelFn = extern "C" fn (TextModelWrapper);
type MakeVectEmbeddingsFn = extern "C" fn (&TextModelWrapper, *const c_char, usize) -> FloatVec;
type DeleteVecFn = extern "C" fn (FloatVec);
type GetLenFn = extern "C" fn (&TextModelWrapper) -> usize;

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
	load_model: TextModelWrapper::load_model,
	delete_model: TextModelWrapper::delete_model,
	make_vect_embeddings: TextModelWrapper::make_vect_embeddings,
	delete_vec: TextModelWrapper::delete_vec,
	get_hidden_size: TextModelWrapper::get_hidden_size,
	get_max_input_size: TextModelWrapper::get_max_input_len,
};

#[no_mangle]
pub extern "C" fn GetLibFuncs() -> *const EmbedLib {
	&LIB
}
