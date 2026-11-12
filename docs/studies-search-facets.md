# Dynamic facets for the study search

Status: built 2026-10-20 (nada-ai `feat/opensearch`, NADA `develop-semantic-ui`). The request and response fields are
part of the study search contract (`studies-search-contract.md`, sections 4, 5.2 and 6.2).

## 1. Problem

The catalog sidebar (countries, regions, collections, data access, data classification, tags, timeseries databases,
user-defined facets, years) comes from about nine MySQL queries in `Search_helper_model` and `Facet_model`. Their counts
are per repository and tab only: a keyword or another filter does not change them, so they do not match `found`. The
browser caches them per (repository, tab).

## 2. What changes

The facet counts come from nada-ai, in the same `POST /studies/search` call that returns the page, and they describe
the search's result set:

- **The keyword and the filters narrow every facet.** "child labour" lists only the countries with matching studies,
  with their counts for that search.
- **A facet ignores its own selection** (disjunctive faceting). With Kenya selected, the country facet still lists
  Uganda and Tanzania with the counts adding them would give; every other facet is narrowed to Kenya. Values within a
  facet are any-of, so this is what lets a user add a second value.
- **No zero counts.** A value is listed only when at least one study of the result has it, selected values included
  (a selected value that matches nothing is cleared from the filter chips, not the sidebar).
- **The active tab applies.** Facets count the studies of the tab (`types`), as today. The tab counts themselves are
  `search_counts_by_type`, unchanged.

The database facets stay for searches nada-ai does not serve: another search engine, the variable view, a search the
NADA driver sends to the database (page size, exact idno), and an outage with `nada_ai_on_outage = database`.

## 3. Contract

### 3.1 Request

`facets` (optional) asks for the counts:

```json
{
  "query": "child labour",
  "filters": {"countries": [12], "types": ["survey"]},
  "facets": {
    "keys": ["countries", "regions", "collections", "form_ids", "data_class_ids", "tags", "databases", "years"],
    "user": ["datakind"]
  }
}
```

| Facet key | Counts the values of | Ignores the filter |
|---|---|---|
| `countries` | country ids | `countries` |
| `regions` | region ids | `regions` |
| `collections` | repository ids (primary or secondary) | `collections` |
| `form_ids` | data access form ids | `form_ids` |
| `data_class_ids` | data classification ids | `data_class_ids` |
| `tags` | tags | `tags` |
| `databases` | timeseries database idnos | `databases` |
| `years` | min and max data collection year | `year_from`, `year_to` |
| `user: [name]` | term ids of user facet `name` | `facets.<name>` |

`repository` (the active scope), `sids`, `created_*` and `types` are never ignored. Omitting `facets`, or sending
empty `keys` and `user`, returns no `facets`. `facets` on an engine without `capabilities.facets` is
`unsupported_capability`. It replaces the reserved `include_facets` boolean.

### 3.2 Response

```json
"facets": {
  "values": {
    "countries": {"buckets": [{"value": "12", "count": 40}, {"value": "35", "count": 7}], "more": false},
    "tags": {"buckets": [{"value": "gender", "count": 3}], "more": false}
  },
  "user": {"datakind": {"buckets": [{"value": "4", "count": 9}], "more": false}},
  "years": {"min": 1990, "max": 2022}
}
```

- `value` is always a string (ids included), `count` is distinct studies. Buckets are ordered by count, then value.
- At most `studies_facet_size` buckets per facet (setting, default 500); `more` is true when values were left out.
  Selected values beyond them are added when they match at least one study; no value is listed with count 0.
- `years.min`/`max` are null when no study in the set has a year.
- nada-ai returns ids only. NADA adds the labels (country names, repository titles, ...) from its reference tables;
  a value NADA no longer knows is not shown.

### 3.3 New filters

Topics and timeseries databases were turned into `sids` lists by NADA (capped at 5,000), and regions into country ids.
They are now indexed and filtered directly, which the region and database facets need:

| Filter | Type | Index field | NADA source |
|---|---|---|---|
| `regions` | int list (max 300) | `regions` | `region_countries` of the study's countries |
| `topics` | int list (max 500) | `topics` | `survey_topics.tid` |
| `databases` | string list (max 100) | `ts_databases` | `timeseries_db_links.db_idno` |

`regions` and `countries` combine with AND, like every other pair of filters. The variable search keeps expanding a
region into country ids on the NADA side (its own contract has no region filter).

## 4. How nada-ai computes the counts

One extra request per search, on the study index (one document per study, so a bucket count is a number of
studies), run in parallel with the page request:

```
size: 0
query.bool.filter: [published, repository, sids, created] + match
aggs:
  f_<key>: filter = (types + every facet filter except <key>)
           aggs: values = terms(<field>, size), selected = filters(one term per selected value)
  f_years: filter = (types + every facet filter except years), aggs: min(years), max(years)
  u_<name>: same as f_<key>, on fq_<name>
```

`match` selects the search's result set before the facet filters:

| Mode | `match` |
|---|---|
| browse | none |
| lexical | the keyword query (filter context: not scored) |
| exact idno | `term idno` |
| semantic | `terms sid` of the semantic studies |
| hybrid | keyword query OR `terms sid` of the semantic studies |

**Limit in semantic and hybrid.** The vector search returns its studies with every filter applied. A study found only
by the vector search that is outside a selected value is never retrieved, so in a facet with a selection the other
values count keyword matches plus the semantic studies that were returned. Every other count is exact. An exact count
would need one more vector search per selected facet.

## 5. NADA

- **Indexing** (`Catalog_search_metadata_extract`): `topics` and `ts_databases` are added to a study's filters.
  Saving a timeseries database that links series sends the linked series again (`Timeseries_db_model`). Existing
  indexes need the studies re-sent (`search_index` sync or a rebuild) before these filters match.
- **Search** (`Catalog_search_semantic_studies`): sends `regions`, `topics` and `databases` as filters, and asks for
  the facets enabled for the tab (`facets_<tab>` setting) when nada-ai reports `capabilities.facets`. The 5,000-study
  limit now applies to the `sid` parameter only.
- **Sidebar** (`Catalog_browse_service`): with counts from nada-ai, the facets are built from them plus labels
  (`Catalog_facet_labels`), in the shape the sidebar already reads, and the database facet queries are skipped. The
  response carries `facet_mode: "dynamic"`, and the browser then asks for facets on every search instead of caching
  them per tab.

## 6. Performance

- Browse and lexical: the facet request runs next to the page request. It does not score and aggregates small
  keyword/integer fields over the matching studies; expected a few ms to low tens of ms for about ten facets.
- Each facet with a selection gets its own filtered aggregation: cost grows with the number of facets that have a
  selection (usually 0-2).
- Semantic and hybrid: the facet request repeats the keyword query; it runs in parallel, so response time grows
  little but OpenSearch does about one more search per catalog request. This is the main cost.
- Keyword facet fields (`tags`, `repositories`, `ts_databases`, `dataset_type`) build their global ordinals at
  refresh (`eager_global_ordinals`), so the first search after an index change does not pay for them.
- MySQL no longer runs the facet queries for searches nada-ai serves. Responses carry the facets on every search
  (non-zero values plus selected ones).

Measure on nada-demo with `include_debug` (`timing_ms` and the request bodies) before tuning.

## 7. Deploy

1. Deploy nada-ai, then re-apply the filter mapping (`ensure_opensearch_filter_facets_mapping`, admin "ensure filter
   indexes") so `topics`, `ts_databases` and the eager ordinals exist on the current indexes.
2. Deploy NADA and re-send every study (full sync) so the study and chunk documents carry `topics` and `ts_databases`.
   Until then a topic or database filter matches nothing.
