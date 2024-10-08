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

	TextModel pEngine = tLib->load_model ("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", 47);

	FloatVec tEmbeddings = tLib->make_vect_embeddings ( &pEngine, text, text_len );

	for (int i = 0; i < tEmbeddings.len; ++i) {
		printf("Embedding [%d]: %f\n", i, tEmbeddings.ptr[i]);
	}

	// Clean up
	tLib->delete_vec ( tEmbeddings );
	tLib->delete_model ( pEngine );

	return 0;
}
