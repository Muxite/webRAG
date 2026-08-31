# Citation ledger: evidence-compiler prior art

Verified 2026-08-31. Every claim below was checked against a fetched primary source
(arXiv `/abs/` or `/html/`), not against search snippets or recall.

**Protocol.** Locate canonical URL → fetch demanding verbatim spans → record quote +
section + URL → anything without a quote is reported UNVERIFIED, never softened.
Search-engine summaries were treated as leads only; one of them was already wrong
(see Discrepancies).

**Headline result: 7 of 7 real, 7 of 7 claims supported.** This batch is clean, against
a historical ~80% real-citation-fabricated-payload rate on this repo's prior sweeps. The
reason is visible in the source text: the claims were written *without* specific numbers.
Vague claims survive verification because they assert less.

---

## 1. FActScore — VERIFIED

- arXiv [2305.14251](https://arxiv.org/abs/2305.14251) · EMNLP 2023
- Min, Krishna, Lyu, Lewis, Yih, Koh, Iyyer, Zettlemoyer, Hajishirzi

Claim: *decompose a response into atomic facts and score the proportion supported by a
reliable source.*

> "we introduce FACTSCORE, a new evaluation that breaks a generation into a series of
> atomic facts and computes the percentage of atomic facts supported by a reliable
> knowledge source." — Abstract

Near word-for-word. Also verbatim: "ChatGPT only achieves 58%"; automated estimator
"with less than a 2% error rate".

**Relevance:** this is the metric shape for the compiler's output. It scores a *finished
answer* against sources — which is what a benchmark scorer needs, not what a runtime
gate needs.

## 2. RAGChecker — VERIFIED (body-level)

- arXiv [2408.08067](https://arxiv.org/abs/2408.08067) · NeurIPS 2024 D&B · Amazon AWS AI et al.

Claim: *separates retrieval quality, generation faithfulness, and end-to-end correctness
via claim-level diagnosis.*

The abstract only says "diagnostic metrics for both the retrieval and generation
modules" — two modules, not three metric families, and it does not mention claims at
all. Both details required fetching the body:

> "we introduce two components: 1) a text-to-claim extractor that decomposes a given
> text T into a set of claims {ci}, and 2) a claim-entailment checker" — §3.2

> "Overall Metrics to provide a holistic view...Diagnostic Retriever Metrics to evaluate
> the effectiveness of the retriever...Diagnostic Generator Metrics to assess the
> performance of the generator" — §3.3

**Relevance: highest of the seven.** This is the answer to "where does a competitor's
lead enter the pipeline" — retrieval, extraction, or synthesis. Directly applicable to
localizing a score gap instead of guessing at it.

**Caveat:** its claim extractor and entailment checker are themselves LLM calls. Adopting
it means adding a judge, and this repo's merge-AUC history is a standing warning about
LLM judges. Use its *decomposition structure*; do not assume its judge transfers.

## 3. Chain-of-Note — VERIFIED, with a caveat the source claim omitted

- arXiv [2311.09210](https://arxiv.org/abs/2311.09210) · EMNLP 2024
- Yu, Zhang, Pan, Ma, Wang, Yu

Claim: *per-document reading notes improve robustness to noisy/irrelevant documents and
improve rejection of out-of-scope questions.*

> "The core idea of CoN is to generate sequential reading notes for retrieved documents,
> enabling a thorough evaluation of their relevance to the given question and integrating
> this information to formulate the final answer." — Abstract

> "CoN achieves an average improvement of +7.9 in EM score given entirely noisy retrieved
> documents and +10.5 in rejection rates for real-time questions that fall outside the
> pre-training knowledge scope." — Abstract

**Omitted caveat — this is a TRAINED method, not a prompting technique:**

> "We employed ChatGPT to create training data for CoN, which was subsequently trained on
> a LLaMa-2 7B model." — Abstract

The +7.9 / +10.5 gains belong to a fine-tuned 7B model. Prompting a stock 7B into writing
reading notes is a *different intervention* with no evidence behind it here. Any
adaptation must be measured locally and must not cite these numbers.

## 4. CRAG — VERIFIED

- arXiv [2401.15884](https://arxiv.org/abs/2401.15884) · Yan, Gu, Zhu, Ling

Claim: *a lightweight retrieval evaluator categorizes retrieval quality and triggers
different retrieval actions.*

> "a lightweight retrieval evaluator is designed to assess the overall quality of
> retrieved documents for a query, returning a confidence degree based on which different
> knowledge retrieval actions can be triggered." — Abstract

Exact. Also verbatim: web search "utilized as an extension for augmenting the retrieval
results"; a "decompose-then-recompose algorithm"; "CRAG is plug-and-play".

**Relevance:** the corrective-control loop — a retrieval state drives a typed next
action. The evaluator being *lightweight and separate* is the transferable part.

**Note:** the search summary attributed a "T5-based evaluator" and a SHAP finding about
named-entity alignment to this paper. Neither is in the abstract. UNVERIFIED — plausible
but not checked against the body.

## 5. Self-RAG — VERIFIED (body-level)

- arXiv [2310.11511](https://arxiv.org/abs/2310.11511) · Asai, Wu, Wang, Sil, Hajishirzi

Claim: *trains a model to emit retrieval and critique tokens for retrieval necessity,
passage relevance, claim support, and usefulness.*

Abstract confirms training and "reflection tokens" but does not enumerate the four types.
From Table 1, "Four types of reflection tokens used in Self-Rag":

| Token | Definition (verbatim) | Outputs |
|---|---|---|
| `Retrieve` | "Decides when to retrieve with ℛ" | yes / no / continue |
| `IsRel` | "d provides useful information to solve x" | relevant / irrelevant |
| `IsSup` | "All of the verification-worthy statement in y is supported by d" | fully / partially / no support |
| `IsUse` | "y is a useful response to x" | 5 / 4 / 3 / 2 / 1 |

**Relevance:** a ready-made state vocabulary. But the gains come from *training* the
model to emit these tokens. A stock local model asked to self-label all four is not
running Self-RAG. Set them from deterministic checks where possible; let the model
propose only where no check can decide.

## 6. RARR — VERIFIED

- arXiv [2210.08726](https://arxiv.org/abs/2210.08726) · ACL 2023
- Gao, Dai, Pasupat, Chen, Chaganty, Fan, Zhao, Lao, Lee, Juan, Guu

Claim: *retrieves attribution for claims and edits unsupported content while preserving
supported content.*

> "RARR (Retrofit Attribution using Research and Revision), a system that 1) automatically
> finds attribution for the output of any text generation model and 2) post-edits the
> output to fix unsupported content while preserving the original output as much as
> possible." — Abstract

Exact. Also verbatim: requires "only a handful of training examples, a large language
model, and standard web search."

**Relevance:** post-hoc repair. Weakest fit of the seven for a system that builds answers
from verified claims forward — by construction there is nothing to retrofit. Useful only
as a fallback audit.

## 7. Program of Thoughts (PoT) — VERIFIED, and the number is stronger than claimed

- arXiv [2211.12588](https://arxiv.org/abs/2211.12588) · Chen, Ma, Wang, Cohen
- Already in this repo: referenced from `docs/research/web/plan-and-solve-prompting.md`

Claim: *an average gain over CoT across mathematical and financial QA evaluations.*

> "To disentangle computation from reasoning, we propose 'Program of Thoughts' (PoT),
> which uses language models (mainly Codex) to express the reasoning process as a program.
> The computation is relegated to an external computer" — Abstract

> "PoT can show an average performance gain over CoT by around 12% across all the
> evaluated datasets." — Abstract

Evaluated on five math word-problem datasets (GSM, AQuA, SVAMP, TabMWP, MultiArith) and
three financial-QA datasets (FinQA, ConvFinQA, TATQA).

**Relevance:** the architectural warrant for a deterministic derivation layer. Model
formulates, interpreter computes. Note "mainly Codex" — the era's model, so the 12% is
not a forecast for a current local 7B.

---

## Discrepancies caught by going to primary source

1. **Chain-of-Note authorship.** A search aggregator listed eight authors, adding
   "Peixin Cao" and "Jian Li". The arXiv `/abs/` page lists six. Unresolved — plausibly a
   version difference between v1 and v2, possibly aggregator error. Cite the arXiv page.
2. **RAGChecker's claim-level mechanism** — the single most useful detail — is absent
   from the abstract. Verifying at abstract depth alone would have marked it UNVERIFIED.
3. **CRAG's T5 evaluator / SHAP finding** appeared only in a search summary. Left
   UNVERIFIED rather than repeated.

## What none of these papers provide

No paper here does deterministic cross-record arithmetic with unit-dimension refusal.
PoT is closest and stops at "run the generated program" — no unit algebra, no evidence
provenance on operands, no abstain path when an operand is missing. That gap is real
prior-art space, not a solved problem to import.
