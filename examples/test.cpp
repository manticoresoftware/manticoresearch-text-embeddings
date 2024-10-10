// clang -O3 -o test test.c -L target/release -lmanticoresearch_text_embeddings
// clang -O3 -o test test.c -I.. -L target/release -lmanticoresearch_text_embeddings
// clang -O3 -o test examples/test.c -I. -L target/release -lmanticoresearch_text_embeddings
#include <stdio.h>
#include <string.h>
#include "manticoresearch_text_embeddings.h"

int main() {
	const char text[] = "This is a sample text.";
	const uintptr_t text_len = sizeof(text);

	const EmbedLib *tLib = GetLibFuncs();

	// Create a new TextEmbeddings instance

	const char modelName[] = "openai/text-embedding-3-small";
	const uintptr_t modelNameLen = sizeof(modelName) - 1;
	const char apiKey[] = "your_api_key_here";
	const uintptr_t apiKeyLen = sizeof(apiKey) - 1;
	TextModelWrapper pEngine = tLib->load_model(modelName, modelNameLen, apiKey, apiKeyLen);

	FloatVec tEmbeddings = tLib->make_vect_embeddings ( &pEngine, text, text_len );

	for (int i = 0; i < tEmbeddings.len; ++i) {
		printf("Embedding [%d]: %f\n", i, tEmbeddings.ptr[i]);
	}

	// Clean up
	tLib->delete_vec ( tEmbeddings );
	tLib->delete_model ( pEngine );

	return 0;
}
