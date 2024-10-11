// Auto-generated file. Do not edit.

#ifndef MANTICORESEARCH_TEXT_EMBEDDINGS_H
#define MANTICORESEARCH_TEXT_EMBEDDINGS_H

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

using TextModelWrapper = void*;

using LoadModelFn = TextModelWrapper(*)(const char*,
                                        uintptr_t,
                                        const char*,
                                        uintptr_t,
                                        const char*,
                                        uintptr_t);

using DeleteModelFn = void(*)(TextModelWrapper);

struct FloatVec {
  const float *ptr;
  uintptr_t len;
  uintptr_t cap;

  bool operator==(const FloatVec& other) const {
    return ptr == other.ptr &&
           len == other.len &&
           cap == other.cap;
  }
  bool operator!=(const FloatVec& other) const {
    return ptr != other.ptr ||
           len != other.len ||
           cap != other.cap;
  }
};

using MakeVectEmbeddingsFn = FloatVec(*)(const TextModelWrapper*, const char*, uintptr_t);

using DeleteVecFn = void(*)(FloatVec);

using GetLenFn = uintptr_t(*)(const TextModelWrapper*);

struct EmbedLib {
  uintptr_t size;
  LoadModelFn load_model;
  DeleteModelFn delete_model;
  MakeVectEmbeddingsFn make_vect_embeddings;
  DeleteVecFn delete_vec;
  GetLenFn get_hidden_size;
  GetLenFn get_max_input_size;

  bool operator==(const EmbedLib& other) const {
    return size == other.size &&
           load_model == other.load_model &&
           delete_model == other.delete_model &&
           make_vect_embeddings == other.make_vect_embeddings &&
           delete_vec == other.delete_vec &&
           get_hidden_size == other.get_hidden_size &&
           get_max_input_size == other.get_max_input_size;
  }
  bool operator!=(const EmbedLib& other) const {
    return size != other.size ||
           load_model != other.load_model ||
           delete_model != other.delete_model ||
           make_vect_embeddings != other.make_vect_embeddings ||
           delete_vec != other.delete_vec ||
           get_hidden_size != other.get_hidden_size ||
           get_max_input_size != other.get_max_input_size;
  }
};

extern "C" {

const EmbedLib *GetLibFuncs();

} // extern "C"

#endif // MANTICORESEARCH_TEXT_EMBEDDINGS_H
