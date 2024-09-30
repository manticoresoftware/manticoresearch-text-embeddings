#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef const void *TextModel;

typedef TextModel (*LoadModelFn)(const char*, uintptr_t);

typedef void (*DeleteModelFn)(TextModel);

typedef struct FloatVec {
  float *ptr;
  uintptr_t len;
  uintptr_t cap;
} FloatVec;

typedef struct FloatVec (*MakeVectEmbeddingsFn)(TextModel, const char*, uintptr_t);

typedef void (*DeleteVecFn)(struct FloatVec);

typedef uintptr_t (*GetLenFn)(TextModel);

typedef struct EmbeddLib {
  uintptr_t size;
  LoadModelFn load_model;
  DeleteModelFn delete_model;
  MakeVectEmbeddingsFn make_vect_embeddings;
  DeleteVecFn delete_vec;
  GetLenFn get_hidden_size;
  GetLenFn get_max_input_size;
} EmbeddLib;

struct EmbeddLib GetLibFuncs(void);
