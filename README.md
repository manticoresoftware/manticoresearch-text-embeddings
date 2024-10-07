# manticoresearch-text-embeddings
Proof of Concept to use Rust in building lib for generating text embeddings


## How to build rust library

```bash
cargo build --lib --release
```

## How to build examples/test.c

```bash
g++ -o test examples/test.c -Ltarget/release -lmanticoresearch_text_embeddings -I. -lpthread -ldl -std=c++11
```

