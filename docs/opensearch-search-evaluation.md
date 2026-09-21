# Study search evaluation (plan step 8)

How the study search was measured and how its defaults were chosen. Reproduce with `eval/run_golden.py`.

## Method

- **Catalog.** The full NADA catalog of the development instance: 1391 published studies (914 documents, 199
  time series, 190 surveys, and smaller types), indexed with the real embedding model
  (`microsoft/harrier-oss-v1-270m`, 640 dimensions) into 1387 study documents and 1932 chunks. Four studies are not
  indexed (see findings). About 65 studies are placeholder or test records; they stay in the index on purpose and count
  as false positives when returned.
- **Golden queries** (`eval/golden_queries.json`, 67 queries): navigational (12), topical (12), acronym (7),
  semantic paraphrase (10), topic plus country filter (5), typo (6), multilingual (4), year (4), document (2) and
  negative (5). The expected studies come from rules over catalog fields (title tokens, or title, abstract and keywords
  for the loosest grade), never from search output. Grades: 3 right answer, 2 clearly relevant, 1 loosely related.
  38 queries are *broad* (over 30 relevant studies); recall is not reported for them.
- **Metrics.** nDCG@10 (gain 2^grade - 1), p@10 (share of the top ten with any relevance), MRR (first result at grade 2 or
  better), recall of the grade 2+ set within the result, *precision of the whole result* (how much of what is returned
  is on topic, the measure of "returns everything"), the median result size, and how many negative queries return nothing.
- The harness calls the same executors as the API against OpenSearch directly, so the ranking policy can be varied per run
  without restarting a server.

## Result at the shipped defaults

| mode | ndcg@10 | p@10 | mrr | recall | precision (whole result) | median size | negatives empty |
|---|---|---|---|---|---|---|---|
| lexical | 0.781 | 0.791 | 0.828 | 0.850 | 0.705 | 17 | 2/5 |
| semantic | 0.754 | 0.736 | 0.844 | 0.805 | 0.590 | 28 | 3/5 |
| hybrid | 0.850 | 0.750 | 0.912 | 0.928 | 0.534 | 46 | 1/5 |

Hybrid is the best mode overall (nDCG@10, MRR, recall) and no positive query returns nothing. By category (nDCG@10):

| category | lexical | semantic | hybrid |
|---|---|---|---|
| navigational | 0.959 | 0.802 | 0.945 |
| topical | 0.937 | 0.796 | 0.978 |
| acronym | 0.696 | 0.810 | 0.844 |
| semantic paraphrase | 0.328 | 0.653 | 0.566 |
| topic + filter | 0.800 | 0.772 | 0.797 |
| typo | 0.807 | 0.470 | 0.749 |
| multilingual | 0.783 | 0.853 | 0.933 |
| year | 0.884 | 0.823 | 0.967 |

Keyword search cannot answer paraphrases (0.33) and semantic search cannot handle typos (0.47); hybrid recovers most of
both, but it does not reach the better single mode on either (paraphrases 0.57 against 0.65; typos 0.75 against 0.81).
That is the cost of one fixed blend.

## What changed because of the measurements

1. **Fuzziness `AUTO` became `AUTO:5,9`.** With `AUTO`, 3 and 4 letter words and two edits on 6 letter words matched
   unrelated words, mostly in the very large `var_keywords` field (`recipe` matched `recibe`, `reside`, `revise`). Keyword
   nDCG rose from 0.768 to 0.820 and a nonsense three-word query dropped from 68 matches to 3. Typo queries did not
   change (0.86). Turning fuzziness off loses typos entirely (0.05).
2. **A stopword analyzer did not help** (nDCG 0.809 against 0.820 for the same query), so the study index keeps its
   simple analyzer.
3. **The semantic floor moved from 0.68 to 0.70.** The 11 study sample suggested real queries peak at 0.70 to 0.85 and
   gibberish at 0.64 to 0.67. On the full catalog the best score of a real query is 0.70 to 0.91, out-of-domain queries
   score 0.68 to 0.69, and **gibberish scores 0.73, above short real queries**. No floor separates gibberish from a
   misspelled one word query with this model, so the floor only rejects clearly unrelated queries.
4. **A relative cutoff on keyword scores was added** (`NADA_STUDIES_LEXICAL_RELATIVE_CUTOFF`, default 0.4). Keyword
   scores are unbounded, so a fraction of the best score removes the tail of partial matches. Without it, hybrid returned
   a median of 100 studies with 17% of them on topic; now the median is 46 with 53% on topic, for a nDCG@10 that stays
   at 0.85 and a recall that drops from 0.976 to 0.928. This is the "return only the best matches" lever.
5. **The semantic cutoff went from 0.90 to 0.94 and the fusion weights to equal (1.0 and 1.0).** Vector scores are
   compressed into a narrow band, so 0.90 of the best barely cuts. With weights 1.0 and 0.5 keyword noise buried the
   paraphrase matches (0.41); with 1.0 and 2.0 paraphrases reach 0.70 but navigational and typo queries fall (0.81 and
   0.66). Equal weights give the best overall nDCG.

The parameter grid behind these choices (floor 0.68 to 0.72, semantic cutoff 0.90 to 0.98, keyword cutoff 0 to 0.7, weights
0.5 to 2.0) is one command: `eval/run_golden.py --sweep`. The chosen point trades recall for precision; a deployment that
prefers recall can lower `NADA_STUDIES_LEXICAL_RELATIVE_CUTOFF` and `NADA_STUDIES_SEMANTIC_RELATIVE_CUTOFF`.

## Findings that are not settings

- **Gibberish is not rejected in hybrid or semantic mode** (it returns the best vector matches above the floor). The
  negative queries `chocolate cake recipe` and `quantum computing hardware` do find real keyword matches, because those
  words occur in variable labels of microdata studies; `bitcoin price prediction` matches price index documents both ways.
  The golden expectation of "nothing" for these is stricter than either leg can meet. Only a different signal (a model that
  scores unrelated text lower, or a reranker) can fix gibberish.
- **Four studies could not be indexed** because the ingest fetches metadata by idno in a URL path and their idnos contain
  a slash, a colon, a space, or are the string `0`. Keying the fetch by `sid` would fix it. NADA-side follow-up.
- **Fourteen studies have no chunks** (empty documents, mostly geospatial), so they are found by keywords only.
- The four experiments that did not pay off: a stopword analyzer, fuzziness `AUTO:4,8`, `minimum_should_match` 75% and 2<85%.

## Limits of this evaluation

- The catalog is a development instance with many placeholder records; results on a curated production catalog will differ,
  and the thresholds should be re-checked there with `eval/run_golden.py`.
- Relevance rules are a proxy for judgement (semantic, multilingual and document queries most of all).
- Page-level passages are not measured: the documents in the catalog carry metadata only.
- 62 positive queries make differences below about 0.01 nDCG noise.
