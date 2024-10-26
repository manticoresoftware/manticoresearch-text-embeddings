// clang -O3 -o test test.c -L target/release -lmanticoresearch_text_embeddings
// clang -O3 -o test test.c -I.. -L target/release -lmanticoresearch_text_embeddings
// clang -O3 -o test examples/test.c -I. -L target/release -lmanticoresearch_text_embeddings
#include <stdio.h>
#include <string.h>
#include <iostream>
#include "manticoresearch_text_embeddings.h"

int main() {
	const char text[] = "This is a sample text.";
	const uintptr_t text_len = sizeof(text);

	const EmbedLib *tLib = GetLibFuncs();

	// Create a new TextEmbeddings instance

	// const char modelName[] = "openai/text-embedding-ada-002";
	const char modelName[] = "sentence-transformers/all-MiniLM-L6-v2";
	const uintptr_t modelNameLen = sizeof(modelName) - 1;
	const char cachePath[] = ".cache/manticore";
	const uintptr_t cachePathLen = sizeof(cachePath) - 1;
	const char apiKey[] = "";
	const uintptr_t apiKeyLen = sizeof(apiKey) - 1;
	const bool useGpu = false;
	TextModelResult pResult = tLib->load_model(
		modelName,
		modelNameLen,
		cachePath,
		cachePathLen,
		apiKey,
		apiKeyLen,
		useGpu
	);
	if (pResult.m_szError) {
		std::cerr << "Error: " << pResult.m_szError << std::endl;
		tLib->free_model_result(pResult);
		return 1;
	}
	TextModelWrapper pEngine = pResult.m_pModel;

	FloatVecResult tVecResult = tLib->make_vect_embeddings ( &pEngine, text, text_len );
	if (tVecResult.m_szError) {
		std::cerr << "Error: " << tVecResult.m_szError << std::endl;
		tLib->free_vec_result(tVecResult);
		return 1;
	}
	FloatVec tEmbeddings = tVecResult.m_tEmbedding;

	for (int i = 0; i < tEmbeddings.len; ++i) {
		printf("Embedding [%d]: %f\n", i, tEmbeddings.ptr[i]);
	}

	// Clean up
	tLib->free_vec_result(tVecResult);
	tLib->free_model_result(pResult);

	return 0;
}
