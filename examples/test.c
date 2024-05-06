// clang -O3 -o test test.c -L target/release -lmanticoresearch_text_embeddings
#include <stdio.h>
#include <string.h>
#include <stdlib.h> // for malloc
#include "manticoresearch_text_embeddings.h"

int main() {
	const char *text = "This is a sample text.";
	uintptr_t text_len = strlen(text);

	// Create a new TextEmbeddings instance
	TextEmbeddings *embeddings = malloc(sizeof(TextEmbeddings));
	if (embeddings == NULL) {
		printf("Failed to allocate memory for TextEmbeddings instance.\n");
		return 1;
	}

	*embeddings = new("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", 47);

	const float *embeddings_ptr = get_text_embeddings(embeddings, text, text_len);
	if (embeddings_ptr == NULL) {
		printf("Failed to get text embeddings.\n");
		return 1;
	}

	const int embeddings_len = get_hidden_size(embeddings);
	for (int i = 0; i < embeddings_len; i++) {
		printf("Embedding [%d]: %f\n", i, embeddings_ptr[i]);
	}

	// Clean up
	free(embeddings);

	return 0;
}
