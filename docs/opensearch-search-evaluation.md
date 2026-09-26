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

## Update: keyword matches are no longer capped

The results above were measured with the result cap of 100, the keyword cutoff of 0.4 and a fusion window of 200 per leg.
That design showed users a tight list (`education` returned 19 studies where the database returns 267) and stopped at 100
results with no notice. It was replaced so that **every keyword match is returned and paged, and only the semantic side is
bounded**. The measurements behind the change, all on the same 67 golden queries and 1391-study catalog:

**1. The keyword score cutoff is a cliff, not a dial.** With the cap and window lifted, the number of keyword matches for a
broad word at different relative cutoffs:

| keyword search | no cutoff | 0.02 | 0.05 | 0.1 | 0.2 | database |
|---|---|---|---|---|---|---|
| `education` | 273 | 271 | 225 | 48 | 11 | 267 |
| `health` | 145 | 145 | 92 | 55 | 33 | 144 |
| `population census` | 212 | 211 | 201 | 166 | 17 | 212 |
| `agriculture` | 165 | 162 | 143 | 103 | 72 | 133 |
| `poverty` | 88 | 77 | 54 | 32 | 25 | 87 |

The 11 `education` studies with the word in the title score 65 to 127, and the other 262 mention it only in low-weight
fields (keywords, variable labels) and score under 25, so any cutoff of 0.2 or more drops all of them at once. Without a
cutoff OpenSearch returns the same studies as the database, ranked with the strongest matches first. On the golden queries
no cutoff scored the same ranking (nDCG@10 0.852 against 0.850 at 0.4), and a floor of 0.02 changed nothing measurable, so
the cutoff was removed. The cost is a larger, less precise tail: precision of the whole result falls from 0.69 to 0.48.

**2. The semantic side needs its own bound.** With the cap lifted, a gibberish query returned 210 semantic-only results;
the cap of 100 was all that limited it. The semantic side is now bounded on its own (`NADA_STUDIES_SEMANTIC_WINDOW`, 50).

**3. Fusing beats pinning.** A first version put the semantic studies in a block above the keyword matches. It scored 0.814
nDCG@10 against 0.850 before: exact-title queries lost most (navigational 0.81 against 0.95), because up to 50 related
studies sat above an exact keyword hit that the vector search had not found (`Geocoded Disasters GDIS dataset` went from
rank 1 to outside the top 10). A block on top works when the keyword ranking is coarse (NADA's database fulltext), not
against OpenSearch's field-boosted scoring. The shipped design fuses the semantic studies with the best keyword matches
by rank, then lists the other keyword matches. The keyword window barely matters (10, 25, 50 and 100 all rank 0.849 to
0.851), so it is 50.

### Result of the shipped design

| mode | nDCG@10 | p@10 | MRR | recall | precision (whole result) | median size | negatives empty |
|---|---|---|---|---|---|---|---|
| lexical | 0.820 | 0.667 | 0.831 | 0.907 | 0.289 | 64 | 2/5 |
| semantic | 0.754 | 0.736 | 0.844 | 0.805 | 0.590 | 28 | 3/5 |
| hybrid | 0.849 | 0.652 | 0.913 | 0.949 | 0.194 | 76 | 1/5 |

Ranking quality is unchanged from the capped design (0.849 against 0.850) and recall is higher (0.949 against 0.928). The
whole-result precision and size are not comparable with the earlier table: the earlier lists were cut, these are not, and
`found` now counts every keyword match. A sweep of the semantic floor (0.68 to 0.72), cutoff (0.90 to 0.98) and window (25,
50, 100) stayed between 0.840 and 0.859 nDCG@10, so the earlier floor and cutoff carry over.

### What users see

Through NADA, `education` returns 276 studies (140 documents, 123 surveys and 13 others), `health` 151 and `population
census` 213, and every page walks with no repeats or gaps and tab counts that add up. A search sorted by title lists the
same set (`found` does not change with the sort). The trade-off is a longer, weaker tail: `Foreign direct investment`
returned 5 studies before and 49 now (the 5 series first, then studies that match some of the words).

### Limits

- Deep paging stops at 10,000 results (`offset + limit`), as before, and `truncated` now says `found` is beyond that depth. A
  request past it is `offset_out_of_range`.
- The semantic side still returns noise for gibberish and for languages the model handles poorly (a Kinyarwanda query for
  foreign direct investment returns 50 Kinyarwanda documents, not the English FDI series, which rank 106th and below): the
  bound keeps it to 50, but the floor and cutoff cannot tell these from real matches. That review is separate.
- Not measured: OpenSearch at 20,000 studies (paging depth, memory and latency).

## Update: field weights and a phrase bonus

**The problem.** `Foreign direct investment` in the survey tab ranked `Informal Survey 2008` first. That study has the phrase
nowhere: its three words appear scattered in unrelated variable labels ("investment support agency", "purchase fixed assets
investments", "private domestic companies foreign nationality owners"), and 78% of its score (57.0 of 72.7) came from the
variable-label field, which was boosted 15. Matching needs only 75% of the words (2 of 3), in any positions of one field,
and nothing rewarded a phrase.

**The change** (`LEXICAL_FIELDS`, `PHRASE_BOOST`, in `studies_search.py`):

| field | before | now |
|---|---|---|
| `idno.text` | 60 | 60 |
| `title` | 40 | 40 |
| `nation` | 30 | 30 |
| `authoring_entity` | 10 | 10 |
| `abstract` | 1 | **10** |
| `keywords` (NADA's combined metadata text) | 10 | **1** |
| `methodology` | 1 | 1 |
| `var_keywords` (variable names, labels, questions) | 15 | **1** |

A query of two or more words also scores the words as a phrase (up to 2 words apart, any field, boost 2). The phrase clause is
optional, so it never changes which studies match, only their order.

**Measured on the 67 golden queries** (the result sizes do not change because matching does not):

| variant | keyword-only nDCG@10 | hybrid nDCG@10 | study 288 in the survey tab |
|---|---|---|---|
| before (abstract 1, keywords 10, variable labels 15) | 0.820 | 0.849 | first |
| abstract 10, keywords 1, variable labels 1 | 0.825 | 0.849 | first |
| abstract 20, keywords 1, variable labels 1 | 0.821 | 0.847 | first |
| abstract 30, keywords 1, variable labels 1 | 0.807 | 0.844 | first |
| **abstract 10, keywords 1, variable labels 1, phrase x2 (shipped)** | **0.840** | **0.852** | second |
| same with phrase x5 | 0.837 | 0.852 | second |
| the old weights with phrase x2 | 0.833 | 0.851 | second |

The weights alone barely move the ranking and do not move study 288: every other survey is an even weaker match, so it stays
first. The phrase bonus is what helps: the study that actually contains the phrase (a survey question about foreign direct
investment, in `World Bank Group Country Survey 2015`) now leads, and 288 follows it. The abstract weight matters little
between 10 and 20 and is worse at 30. A bonus of 2 and 5 rank the same. The golden queries measure the ranking of well-known
items, so the gain from the weights is within noise; the phrase bonus is the measurable improvement (keyword-only 0.820 to
0.840).

**Limits.** A study that mentions the words in unrelated places still matches (and is listed after the ones that do not), because
matching itself is unchanged: 75% of the words, anywhere in a field. Requiring every word, or a stricter rule for short
queries, is a separate decision that would change what matches.

## Update: queries with no searchable content are rejected before they reach the engine

A wider edge-query review (43 queries: gibberish, absent-domain, very short, stopword-only, operators, fake
identifiers) found a regression from uncapping the keyword matches: `the and of` returned 429 studies, `12` returned
172, `a b c` and `' OR 1=1 --` similarly high. `minimum_should_match: 2<75%` only needs a fraction of a query's words,
and short or common words occur almost everywhere in the long `keywords`/`var_keywords` fields, so with no score
cutoff nothing bounded that tail.

**An absolute score floor does not work**, for the same reason a relative one did not: real and noise matches occupy
the same range. The lowest score of a genuinely relevant hit (`povery`, a fuzzy typo match) is 0.149; the noise query
`x` scores down to 0.19. No threshold separates them.

**Is this an OpenSearch configuration gap?** The `nada_text` analyzer used on every text field has no stopword
filter (confirmed directly: analyzing `"the and of"` returns all three tokens unchanged), so this is a lexical
configuration matter, not a semantic one. Adding a `stop` filter was tested on a copy of the index: it does make
OpenSearch return 0 for `the and of` (an all-stopword query analyzes to nothing, and a query with nothing left
matches nothing), but it costs real ranking quality (golden nDCG@10 0.840 → 0.824, matching an earlier finding from
before the keyword cutoff was removed) and does not touch non-stopword noise (`12`, `ab`, `x` are unaffected; `a b c`
got worse, 16 → 87, because removing "a" changes how many terms `minimum_should_match` requires). Stopwords and short
tokens are two different problems, and an analyzer only addresses one of them, at a cost.

**The fix that shipped, briefly**: `has_searchable_content()` checked the query text before either leg ran, dropping
stopwords and tokens under 3 characters, and rejecting a query with nothing left. It fixed `the and of`, `12`, `ab` and
similar noise, with zero false positives on the golden queries. It was superseded (see below) once it turned out to
break a related, more important case.

## Update: an exact idno match runs first instead, and the noise-word gate is gone

A single-token query is very often someone pasting an idno. The database's own search has always checked this first
(`Catalog_study_idno_lookup`): an exact, case-insensitive match on `surveys.idno` (and its aliases) answers the search
by itself. `/studies/search` never had an equivalent, and relying on the scored `idno.text^60` field alone is not
precise: it is tokenized on `_` and `-`, so `AGO_2020_HRPM_GEO_v01_M` splits into six fragments and a 75% match pulls
in unrelated studies that merely share `geo`, `v01` or `2020` — the real study was returned third, in a result of 51.

Testing the noise-word gate against this uncovered a real bug: the same word-splitting the gate does to decide whether
a query has "content" fragments a compact idno like `PC11_A02-28-v22` into `pc`, `11`, `a`, `02`, `28`, `v`, `22` —
every piece under 3 characters — and the gate rejected the whole query, so an exact, valid idno search returned
nothing.

**What shipped instead of both:**

- `exact_idno_match()` (`search/backend/opensearch/studies_search.py`): for any single-token query, in every mode,
  before any scored search runs, a `term` query checks the study index's `idno` field (within the active filters)
  for an exact match. A match is the whole result — the same guarantee the database gives on every other engine —
  `matched_by: ["idno"]`, no score.
- The `idno` field's mapping now has `normalizer: "nada_sort"` (the same lowercase + accent-fold normalizer already
  used for `title_sort`/`nation_sort`), so the match is case- and accent-insensitive without fragmenting the value the
  way the analyzed `idno.text` field does: `ago_2020_hrpm_geo_v01_m` now matches the stored `AGO_2020_HRPM_GEO_v01_M`,
  exactly, as one token. This needs the study index to be rebuilt for the normalizer to apply to already-indexed idnos
  (a mapping change is not retroactive); a `_reindex` into the same mapping is enough, no re-embedding needed.
- The noise-word gate (`has_searchable_content`, `app/studies_query.py`, the `no_searchable_terms` warning) was removed
  entirely, not patched, because the underlying tokenization problem it had is the same one that made `idno.text`
  imprecise in the first place, and patching it to exempt idno-shaped tokens would leave two overlapping heuristics
  for what is really one problem (word-splitting long alphanumeric identifiers).

**Consequence, stated plainly:** removing the gate brings back the stopword/short-token regression it fixed — `the and
of`, `12`, `ab` and similar noise queries are unbounded again, exactly as documented further up this page. The idno fix
does not touch that case at all (those tokens are not idnos, so the exact-match check simply finds nothing and falls
through to the normal search). This is a known, accepted trade-off, not an oversight: if both problems need solving at
once, the noise-word check would need to run only when the idno check does not match, on a query that also is not a
single alphanumeric-with-separators token (so it never re-fragments an idno).

**What is not covered:** aliases (`survey_aliases`) are not indexed anywhere in OpenSearch, so a study findable by its
alias on every other engine remains unfindable here; that still needs a database-side lookup if it is wanted, the way
`qdrant_db`'s driver already does it.

## Update: a title naming every query word is promoted ahead of the fused order

Hybrid's rank fusion only counts a study's position in each leg, not how decisive a match is. `high resolution
angola` found study 234 ("High Resolution Poverty Map (Geospatial Data), Angola, 2020" — every query word is in the
title) ranked 5th: four studies the semantic leg also happened to return outranked it, each counted equally by RRF
regardless of how weak their own relevance was.

**Alternatives tried and rejected** (measured on the golden queries): weighting the keyword leg higher (1.5-2x) did
not move study 234 at all, and cost the paraphrase category (0.54 -> 0.39). A margin-based guard ("keep the keyword
leg's top result first when it leads the runner-up by 20%") worked for this case (nDCG 0.857) but fired on 22 of 67
queries and depends on OpenSearch's raw scores, which shift with any boost or phrase-bonus tuning.

**What shipped**: `_title_is_complete_match()` checks, for each of the keyword leg's own candidates (already
fetched for fusion, so no extra request), whether every real word of the query (stopwords aside) appears somewhere
in that study's title. Any that qualify are moved to the front of the fused list, in their keyword rank order;
everything else keeps the normal fused order behind them. It depends only on the words themselves, not on scores, so
it is unaffected by later boost or weight tuning. Only the relevance-sort path is affected; another sort's union
query and plain `lexical` mode are untouched (`lexical` already ranked study 234 first on its own: a title match
scores far above anything else, boost 40 plus the phrase bonus).

Measured: hybrid nDCG@10 0.852 -> 0.869, MRR 0.919 -> 0.927, p@10 0.665 -> 0.669; recall and the negative-query pass
rate unchanged. `matched_by` is unaffected (a promoted study keeps whatever it already had, `["lexical"]` or both);
this is a pure reordering, not a new kind of match.

## Update: the title promotion needs three real words, and ignores accents

With one word, "every query word is in the title" is just "the word is in the title": 163 of the local index's 1,383
titles contain `census`, so the whole keyword head was promoted and every study only the semantic leg found was pushed
off the first pages — hybrid behaved like keyword search on broad one-word queries. The promotion now needs a query of
at least three real words: study titles are long, and two words in one single nothing out either. The title check also strips accents now, as the index's `nada_text` analyzer does (`cote
d'ivoire` matches a "Côte d'Ivoire" title). Neither stems: the index does not stem either (`surveys` matches 1 title,
`survey` 260), so stemming is an index-wide decision, not this rule's.

**Alternative tried and rejected**: promoting only when at most 5 (or 10) studies qualify, so a broad title match is
not promoted at all. It cost hybrid nDCG@10 0.839 -> 0.828 (0.831 with 10): for `population census` (1.000 -> 0.697),
`census 2011` and `gender statistics profile 2013` the many matching titles are the relevant studies.

Measured (local index, 67 golden queries): hybrid nDCG@10 0.839 -> 0.832, p@10 0.672 -> 0.667, MRR, recall and the
negative-query pass rate unchanged. The cost is two 2-word queries whose matching titles are the relevant studies:
`population census` (1.000 -> 0.697) and `census 2011` (1.000 -> 0.915) — accepted, since a minimum of two would also
promote every title of a broad 2-word query the golden set does not cover well (`household survey`). A minimum of two
measured 0.838. Of the one-word queries, `DHS` fell (1.000 -> 0.934) and `MICS` rose (0.910 -> 1.000).
