# QuPath attention review (QuPath 0.7.0)

Turns a train.py run's `attention.h5` into a blinded QuPath review package for a
pathologist, then scores their annotations against the model's attention.

Self-contained: reads only a run directory (`attention.h5`,
`external_predictions.csv`), `metadata/cohort_labels.csv`, and the original
`.svs` files. Imports nothing from `src/prediction/`. Settings live in `config.py`.

## Workflow

```bash
RUN=../../results/tcga2chile_3class_msiw3_s0_46204792
python select_slides.py  --run-dir $RUN                    # review/review_set.csv + KEY_unblinding.csv
python export_geojson.py --run-dir $RUN                    # review/package/overlays/R01.geojson ...
python build_package.py  --run-dir $RUN \
       --slides-dir <svs folder> --copy-slides             # review/package/ (run in a job: copies GB)
#  -> send review/package/ ; pathologist follows package/README.md
python analyze_annotations.py --run-dir $RUN \
       --annotations <returned>/annotations --form <returned>/review_form.csv
```

Everything goes under `<run-dir>/review/`:

```
review/
  review_set.csv          code -> slide_id, svs name          (ours)
  KEY_unblinding.csv      code -> label, prob, group, subtype (ours, never shipped)
  overlays_manifest.csv   tiles per slide, tile_score limit   (ours)
  package/                <- ships to the pathologist
    README.md  review_form.csv  overlays/  scripts/  slides/
  results/                <- analyze_annotations.py output
```

## Modules

| file | role |
|---|---|
| `config.py` | sampling plan, annotation classes and colours, attention bands, limits |
| `scaling.py` | attention percentile / log2-relative, shared tile-score limit, bands |
| `select_slides.py` | stratified, seeded, blinded review set from one run |
| `export_geojson.py` | attention.h5 -> one QuPath GeoJSON per slide |
| `build_package.py` | render scripts + README + form, optionally copy/rename slides, check blinding |
| `groovy/*.groovy` | templates rendered into the package: import overlays, export annotations |
| `analyze_annotations.py` | annotations x top-attention tiles, per slide and per group |
| `templates/` | pathologist README and review form |

## Overlay format (verified against QuPath 0.7.0 source / docs)

Each tile is a GeoJSON Feature with a square Polygon in level-0 pixels and
`properties = {objectType: "tile", isLocked: true, classification: {name, color: [r,g,b]},
measurements: {attn_pct, attn_rel_log2, tile_score}}`. Import uses
`importObjectsFromFile`, export uses `exportObjectsToGeoJson(..., "FEATURE_COLLECTION")`;
"Run for project" saves each image, "(without save)" does not.

| measurement | definition |
|---|---|
| `attn_pct` | within-slide percentile rank of attention, 0-100 |
| `attn_rel_log2` | log2(attention x n_tiles), clipped to [-2, 4]; 0 = uniform share |
| `tile_score` | EBV log-odds of each tile alone, clipped to +/- a limit shared by all slides |
