# profile-rag

Answers questions about Aditya Chunduri's public profile with retrieval only: no LLM, no API key.

The corpus is the portfolio's own `content.ts`, exported as 318 fact sized chunks by
`npm run export:corpus` in the Portfolio repo. Retrieval is BM25 plus a dense index
(Chroma, `BAAI/bge-small-en-v1.5` via fastembed) fused by reciprocal rank, reranked by a
cross encoder (`Xenova/ms-marco-MiniLM-L-6-v2`), then the best one to three sentences of
the top chunk are returned with links back to the site. A small `faq.yaml` answers the
canonical questions verbatim. Everything runs on CPU with ONNX models.

## Run

```bash
uv sync
uv run python -m profile_rag.build          # index corpus/portfolio.jsonl into data/
uv run uvicorn profile_rag.api:app --reload
curl -s localhost:8000/health
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question":"what is Agent Shield"}'
```

## Ask from the terminal

```bash
uv run python -m profile_rag.ask "what is Agent Shield"
uv run python -m profile_rag.ask --mode dense "retinal segmentation"   # raw vector search, no FAQ or rerank
```

## Check

```bash
uv run pytest                               # builds the index in a subprocess first
uv run python -m profile_rag.eval           # recall@1/3 and MRR per retrieval mode
```

The eval bank in `eval/questions.jsonl` is frozen; `tests/test_retrieval.py` ratchets
hybrid plus rerank recall@3 at a floor set above the rerank ablated value.

## API

- `GET /health` returns chunk count and the corpus sha256, so a deployment can be tied to a content.ts commit.
- `GET /suggestions` returns the FAQ questions.
- `POST /ask {"question"}` returns `{answer, mode: faq|extract|fallback, score, sources[], latency_ms}`.
  Questions are capped at 500 characters and 30 requests per minute per IP. CORS allows the portfolio origin and localhost.

## Deploy

`Dockerfile` builds the index and warms the model caches into the image. Any host that runs a container works;
the portfolio reads the URL from `VITE_PROFILE_CHAT_URL` at build time.
