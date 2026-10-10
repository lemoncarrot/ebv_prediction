# EBV attention review: instructions

{{N_SLIDES}} slides, named R01, R02, ... so the diagnosis is hidden. For each
slide you will (1) make your own EBV call, (2) look at where the model looked,
(3) mark the regions that stand out, and (4) fill one row of `review_form.csv`.

## What is in this folder

| item | what it is |
|---|---|
| `slides/` | the whole-slide images (R01.svs, ...) |
| `overlays/` | the model's attention, one file per slide |
| `scripts/` | two QuPath scripts: import the overlays, export your annotations |
| `review_form.csv` | one row per slide, to fill in |

## Setup (once, QuPath {{QUPATH_VERSION}})

1. Inside this folder, create an empty folder called `project`.
2. QuPath: **File > Project... > Create project** and choose that `project` folder.
3. Drag every file from `slides/` into QuPath and click **Import**.
4. **Automate > Script editor**, then in the editor **File > Open...** `scripts/import_overlays.groovy`.
   Choose **Run > Run for project**, move all images to the right, click **OK**.
5. The annotation classes below should now be in the class list of the
   **Annotations** tab. If they are not, close and reopen the project.

## For each slide

1. **Overlay off, your call first.** Press **D** to hide the overlay (View > Show detections).
   Look at the H&E and record your EBV call and confidence in `review_form.csv`.
2. **Overlay on.** Press **D** again; press **F** to fill the tiles (View > Fill detections).
   Use the opacity slider in the toolbar so the tissue stays visible.
   - Default colours are attention bands: {{BANDS}}.
     To hide the grey band, right-click "Attention rest" in the class list and hide it.
   - For a continuous map: **Measure > Show measurement maps**, then pick:
     - `attn_pct`: where the model looked most **on this slide** (0-100; every
       slide has a top 5%, even if the model's attention was spread evenly)
     - `attn_rel_log2`: how concentrated attention is (0 = even share, 3 = 8x)
     - `tile_score`: whether a region looks EBV-like to the model (+) or not (−).
       Set the colour range to −{{TILE_SCORE_LIMIT}} to +{{TILE_SCORE_LIMIT}}
       so 0 sits in the middle; the range is the same for every slide.
3. **Annotate** the regions that stand out (no need to cover the whole slide),
   using only these classes:
{{ANNOTATION_CLASSES}}
   Draw with the brush or polygon tool, select the class in the list, then **Set class**.
4. Finish the form row, then **File > Save** before moving to the next slide.

## When done

1. Script editor: open `scripts/export_annotations.groovy`,
   **Run > Run for project (without save)**, all images, **OK**.
2. Send back the new `annotations/` folder and `review_form.csv`.
