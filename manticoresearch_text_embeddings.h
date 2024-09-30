#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

using TextModel = const void*;

using LoadModelFn = TextModel(*)(const char*, uintptr_t);

using DeleteModelFn = void(*)(TextModel);

struct FloatVec {
  float *ptr;
  uintptr_t len;
  uintptr_t cap;
};

using MakeVectEmbeddingsFn = FloatVec(*)(TextModel, const char*, uintptr_t);

using DeleteVecFn = void(*)(FloatVec);

using GetLenFn = uintptr_t(*)(TextModel);

struct EmbeddLib {
  uintptr_t size;
  LoadModelFn load_model;
  DeleteModelFn delete_model;
  MakeVectEmbeddingsFn make_vect_embeddings;
  DeleteVecFn delete_vec;
  GetLenFn get_hidden_size;
  GetLenFn get_max_input_size;
};

extern "C" {

EmbeddLib GetLibFuncs();

} // extern "C"
