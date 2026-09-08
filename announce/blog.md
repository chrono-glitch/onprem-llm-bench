# Blog post — nostalgiasistemas.com

The blog post **is** `WRITEUP.es.md` (and `WRITEUP.en.md` for the EN version).
They're written as standalone articles — don't rewrite them, just wrap them.

## Publish steps

1. Copy `WRITEUP.es.md` body into the blog engine as a new post.
   - **Title:** Servir LLMs on-premise sin GPU: los números y cómo elegir
   - **Slug:** `servir-llms-on-premise-sin-gpu`
   - **Summary / meta description:** Benchmark de 7 modelos LLM abiertos (2–8B) en
     una máquina de CPU sin GPU: qué cuantización, qué modelo, qué motor, y si
     vale la pena una GPU. Con guía de decisión.
   - **Tags:** LLM, inferencia on-premise, MLOps, edge AI, llama.cpp, Ollama
2. Fix relative links: `RESULTS.md`, `DECISION.md`, `results/cpu_all.png` etc.
   must point at the GitHub repo
   (`https://github.com/chrono-glitch/onprem-llm-bench/blob/main/...`) or be
   inlined.
3. Embed `results/cpu_all.png` near "Full results".
4. Add a one-line footer: *Código y datos completos:
   github.com/chrono-glitch/onprem-llm-bench*
5. Repeat for `WRITEUP.en.md` as `/en/serving-llms-on-prem-no-gpu` (or wherever
   the EN section of the site lives).
6. Publish blog FIRST, then post LinkedIn + X pointing at the repo (not the blog —
   the repo is the stronger portfolio surface; the blog link can go in the
   LinkedIn first comment too).

## Order of operations (all in one sitting — this is the "finish" step)

- [ ] Blog post ES live
- [ ] Blog post EN live (optional, can follow next day)
- [ ] LinkedIn post published + repo link in first comment
- [ ] X thread posted + pinned
- [ ] Paste the 4 URLs into `~/history/HISTORY.md` and
      `~/dev/onprem-llm-bench/NEXT.md` (check the last two boxes)
