# Dataset Sources and Licenses

## Listings Dataset (listings.csv)

**Size:** 3,000 rows
**Source:** Synthetic generation based on seed data patterns
**License:** MIT (generated for demonstration)
**Provenance:**
- Patterns derived from existing seed data in `backend/data/seed.js`
- Categories, crafts, regions, materials based on Indian handicraft knowledge
- Price distributions based on category complexity multipliers
- Multilingual text samples in English, Hindi, Marathi, Tamil, Telugu, Bengali, Gujarati

**Limitations:**
- Prices are synthetic estimates, not real market data
- Descriptions are template-based, not artisan-written
- For demonstration/prototype use only

## Heritage Corpus (heritage_corpus.jsonl)

**Size:** 30 records
**Source:** Seed data + cultural knowledge
**License:** MIT (generated for demonstration)
**Provenance:**
- 3 records from existing in-memory heritage data in `server.js`
- 27 additional crafts based on Indian GI tags and cultural knowledge
- Historical context based on publicly available cultural information

**Limitations:**
- Historical timelines are simplified
- Cultural contexts are generalized
- For demonstration/prototype use only

## Graph Data (graph_nodes.csv, graph_edges.csv)

**Size:** Derived from listings
**Source:** Synthetic listings dataset
**License:** MIT (generated for demonstration)
**Provenance:**
- Nodes extracted from listings (creators, crafts, materials, regions)
- Edges represent relationships (PRACTISES, USES_MATERIAL, ORIGINATED_IN)

**Limitations:**
- Graph is synthetic, not based on real collaboration data
- For demonstration/prototype use only

## Barter Pairs (barter_pairs.csv)

**Size:** 300 rows
**Source:** Synthetic generation
**License:** MIT (generated for demonstration)
**Provenance:**
- Skills extracted from materials, techniques, and crafts
- Match labels based on complementary skill heuristics

**Limitations:**
- Match labels are heuristic-based, not real user data
- For demonstration/prototype use only

## Speech Samples (speech_samples/)

**Size:** Not included (placeholder)
**Source:** To be sourced from:
- Common Voice (Mozilla) - CC0 license
- IndicVoices - CC BY license
- Or synthesized via TTS for demonstration

**Note:** Currently not included due to size. Use TTS synthesis for demo.

## Images (images/)

**Size:** Not included (placeholder)
**Source:** To be sourced from:
- Wikimedia Commons (CC BY/CC0)
- Unsplash (free for commercial use)
- Government open data (Ministry of Textiles, DC Handicrafts)

**Note:** Currently not included due to size. Use placeholder URLs for demo.

---

## Overall Limitations

All datasets are generated for demonstration and prototype purposes. They are not:
- Real market data
- Real artisan profiles
- Real transaction histories
- Suitable for production machine learning

For production deployment, datasets should be:
- Collected from real platform usage
- Properly licensed and consented
- Curated by domain experts
- Validated by artisans and cultural experts

## Data Split

All datasets use train/val/test split of 70/15/15 with fixed seed (42).
Stratified by category where applicable.
