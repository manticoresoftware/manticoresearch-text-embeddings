// clang -O3 -o test test.c -L target/release -lmanticoresearch_text_embeddings
// clang -O3 -o test test.c -I.. -L target/release -lmanticoresearch_text_embeddings
// clang -O3 -o test examples/test.c -I. -L target/release -lmanticoresearch_text_embeddings
#include <stdio.h>
#include <string.h>
#include <iostream>
#include "manticoresearch_text_embeddings.h"

int main() {
	std::vector<std::string> strings = {"Example text", "Hello world!"};
	std::vector<StringItem> text_list;
	for (const auto& s : strings) {
		text_list.push_back({s.c_str(), s.length()});
	}

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
	printf("Model loaded successfully\n");

	FloatVecResult tVecResult = tLib->make_vect_embeddings ( &pEngine, text_list.data(), text_list.size() );

	if (tVecResult.m_szError) {
		std::cerr << "Error: " << tVecResult.m_szError << std::endl;
		return 1;
	}

	printf("Embeddings computed successfully\n");
	for (size_t i = 0; i < tVecResult.len; ++i) {
		printf("Embeddings for text %zu\n", i);
		const FloatVec& tVec = tVecResult.m_tEmbedding[i];
		printf("Vector size: %zu\n", tVec.len);

		printf("Iterating over embeddings\n");
		for (int j = 0; j < tVec.len; ++j) {
			printf("Embedding [%d]: %f\n", j, tVec.ptr[j]);
		}
	}

	// Clean up
	tLib->free_vec_result(tVecResult);
	tLib->free_model_result(pResult);

	return 0;
}
