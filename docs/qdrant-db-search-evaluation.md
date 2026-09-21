# Qdrant + catalog database search evaluation

NADA's `qdrant_db` semantic engine fuses the catalog database's keyword ranking with Qdrant's vector ranking (see
`Catalog_search_semantic_fused.php` in NADA). This records how it was measured and how its defaults were chosen.
Reproduce with `eval/run_nada_api.py`, which scores NADA's public catalog API, so it works for any search engine.

## Method

- **Catalog and queries.** The same 1391 published studies and 67 golden queries as the OpenSearch evaluation
  (`docs/opensearch-search-evaluation.md`). Qdrant was indexed with the branch code so every point carries
  `metadata.sid`, using a scratch collection.
- **What is measured.** The first 100 studies NADA returns for `GET /api/catalog?sk=...&ps=100` (with `country=` for the
  filtered queries), scored with the same metrics as before. "Median size" is `found`, the size of the whole result;
  "precision (whole result)" is measured over the 100 studies fetched.
- **Configurations compared.** `database` (the catalog database search alone), `legacy_qdrant` (the original Qdrant
  driver), `qdrant_db` (the fused driver at its defaults) and `opensearch` (nada-ai's study search).

## Result

| configuration | nDCG@10 | p@10 | MRR | recall | precision (whole result) | median size | negatives empty |
|---|---|---|---|---|---|---|---|
| database | 0.463 | 0.498 | 0.497 | 0.621 | 0.483 | 12 | 5/5 |
| legacy_qdrant | 0.767 | 0.621 | 0.837 | 0.887 | 0.052 | 1391 | 0/5 |
| qdrant_db | 0.764 | 0.683 | 0.840 | 0.879 | 0.417 | 41 | 3/5 |
| opensearch | 0.850 | 0.750 | 0.912 | 0.928 | 0.534 | 46 | 1/5 |

By category (nDCG@10):

| category | database | legacy_qdrant | qdrant_db | opensearch |
|---|---|---|---|---|
| acronym | 0.60 | 0.84 | 0.92 | 0.84 |
| document | 0.31 | 1.00 | 1.00 | 1.00 |
| multilingual | 0.70 | 0.87 | 0.91 | 0.93 |
| navigational | 0.80 | 0.81 | 0.84 | 0.95 |
| semantic paraphrase | 0.12 | 0.71 | 0.63 | 0.57 |
| topic + filter | 0.71 | 0.93 | 0.93 | 0.80 |
| topical | 0.30 | 0.80 | 0.74 | 0.98 |
| typo | 0.10 | 0.32 | 0.28 | 0.75 |
| year | 0.67 | 0.78 | 0.93 | 0.97 |

- The fused driver ranks as well as the original Qdrant driver (0.764 against 0.767) but returns a bounded result
  (median 41 studies, 42% of the 100 fetched on topic) instead of the whole catalog (1391, 5%). Its `found` and tab counts
  describe that result.
- The database alone answers exact keywords precisely but returns nothing for 6 of the 62 positive queries (paraphrases,
  other languages) and cannot handle typos.
- OpenSearch remains better overall (0.850). Its advantage is keyword quality: the database's fulltext has no fuzziness,
  no field boosts and no stemming, which shows in the typo (0.28 against 0.75) and topical (0.74 against 0.98) categories.
  The fused driver is ahead on acronyms, paraphrases and filtered queries.
- Qdrant's raw cosine scores separate unrelated queries better than OpenSearch's compressed scores: the best hit of a real
  query scored 0.46 to 0.83 (median 0.58) and that of an unrelated query 0.38 to 0.56 (median 0.45). A floor of 0.45 turns
  away most of the negative queries (3 of 5 return nothing; the other two return 6 and 1 studies).

## How the defaults were chosen

Sweep over the floor (none, 0.45, 0.50), the relative cutoff (none, 0.85, 0.90), the keyword weight (1.0, 0.5, 0.3), the
keyword window (100, 500) and the Qdrant window (50, 100), all with the vector weight fixed at 1.0.

| change from no floor and no cutoff | nDCG@10 | median size | precision (whole result) |
|---|---|---|---|
| none (weights 1.0/1.0) | 0.763 | 55 | 0.09 |
| floor 0.45 | 0.747 | 52 | 0.16 |
| cutoff 0.85 | 0.759 | 44 | 0.43 |
| floor 0.45 and cutoff 0.85 | 0.749 | 41 | 0.42 |
| ... and keyword weight 0.5 (shipped) | 0.764 | 41 | 0.42 |
| ... and floor 0.50 | 0.766 | 30 | 0.49 |
| ... and cutoff 0.90 | 0.755 | 35 | 0.49 |

The shipped defaults are floor 0.45, cutoff 0.85, keyword weight 0.5 and vector weight 1.0. The keyword window (100
against 500) and the Qdrant window (50 against 100) made no difference on this catalog, because the cutoff keeps the
semantic side under 50 and few queries have more than 100 keyword matches; the windows stay small (100 and 50) to bound the
work. A floor of 0.50 gives a smaller, more precise result at the price of one positive query returning nothing.

## Pitfalls when running the evaluation

- **A search rate limit answers 429, and the driver treats that as an outage.** nada-ai limits `/search` to 120 requests
  a minute by default; a sweep of 67 queries per run trips it and the runs silently degrade to keyword-only results. Run the
  evaluated nada-ai with `NADA_RATE_LIMIT_SEARCH_PER_MINUTE=0`.
- **PHP's opcache keeps the previous version of a config file for about two seconds.** When a run changes the NADA config
  (engine, URL, thresholds), pause a few seconds before the first request; otherwise the first queries of the run are
  answered under the previous configuration. This was noticed because the first twelve queries of one run returned
  `found: 1391`.

## Limits

- One development catalog with many placeholder records and 62 positive queries: differences below about 0.02 nDCG are
  noise. Re-run on your own catalog before relying on the thresholds.
- The relevance rules are a proxy for judgement, as in the OpenSearch evaluation.
- Document passages (page-level results) are not measured; the catalog's documents carry metadata only.
- The SQL Server version of the keyword leg is unverified: there is no SQL Server in the development environment.
