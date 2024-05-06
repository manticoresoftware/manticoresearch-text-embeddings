#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef void *ModelPtr;

typedef struct TextEmbeddings {
  ModelPtr model;
} TextEmbeddings;

struct TextEmbeddings new(const char *name_ptr, uintptr_t name_len);

const float *get_text_embeddings(const struct TextEmbeddings *self,
                                 const char *text_ptr,
                                 uintptr_t text_len);

uintptr_t get_hidden_size(struct TextEmbeddings *self);

uintptr_t get_max_input_len(struct TextEmbeddings *self);
