# FoodLens — Codebase Reference

Read-only snapshot of the working tree as of 2026-09-17. Line numbers point at
the files as they are now, uncommitted changes included. The only file created
was this one; nothing else was changed.

---

## Part 1 — Files and how they connect

### Dependency flow

```
                          ┌──────────────────────────────────────────────┐
  BROWSER                 │ templates/index.html + static/app.js + .css  │
                          └───────────────┬──────────────────────────────┘
                                          │ fetch() → /api/...
  FRONT ENDS      server.py (Flask, current)          app.py (Streamlit, fallback)
                   │  │  │  │  │  │                     │   │   │
                   │  │  │  │  │  └─ gemini_chat        │   │   └─ charts
                   │  │  │  │  └──── evaluate ──┐       │   └───── predict
                   │  │  │  └─────── dataset_nutrition│       └───────── pipeline
                   │  │  └────────── charts         │
                   │  └───────────── predict ◄──────┘  (evaluate imports predict)
                   └──────────────── pipeline
                                        │
  PIPELINE (src/)             pipeline.py
                          ┌─────────┼──────────┐
                       detect.py  predict.py  nutrition.py ◄── dataset_nutrition.py
                     (YOLOv8)   (MobileNetV3)   (USDA API)

  OFFLINE SCRIPTS (not imported by the app)
    prepare_data.py ──writes──► outputs/subset_meta.json ──read by──► train_classifier.py
    train_classifier.py ──writes──► models/classifier_best.pt, outputs/history.csv,
                                     outputs/split_indices.json
    evaluate.py ──imports predict── writes outputs/figures/confusion_matrix.png
    quick_test.py ──imports predict
    verify_setup.py (standalone)
```

Leaf modules that import no other project file: `detect`, `predict`,
`nutrition`, `charts`, `gemini_chat`, `prepare_data`, `train_classifier`,
`verify_setup`.

`train_classifier.py` imports nothing from the project. It links to the rest
only through files: it reads `subset_meta.json` and writes the checkpoint, which
`predict.load_model` then reads.

---

### Root Python files

#### `server.py` — ~452 lines
**Purpose:** Flask back end that serves the HTML front end and exposes the
pipeline, charts, evaluation, dataset nutrition and Gemini chat as JSON/PNG
endpoints.

| Function / route | What it does |
|---|---|
| `init_model()` | Loads the classifier once per process into globals `MODEL`, `CLASSES`, `DEVICE`. |
| `png_response()` | Wraps PNG bytes in a no-store `Response`. |
| `_payload()` | Reads chart parameters from a JSON POST body or GET query string. |
| `_cache_is_fresh()` / `_read_cache()` | Per-class accuracy cache is valid only if its stored checkpoint mtime matches. |
| `per_class_data()` | Returns the cached per-class report, or runs a full test-set evaluation under a lock and caches it. |
| `GET /` `index()` | Renders `index.html` with `daily_goal` and `conf_threshold`. |
| `POST /api/analyze` | Saves the upload to a temp file, runs `run_pipeline`, annotates the image, multiplies nutrition by count, returns JSON. |
| `/api/chart/macro-pie` | Macro donut chart PNG. |
| `/api/chart/calorie-bar` | Per-item calorie bars. Divides count back out, because `make_calorie_bar` re-applies it. |
| `GET /api/training-runs` | Splits `history.csv` into runs; returns loss and accuracy charts as base64. |
| `GET /api/confusion-matrix` | Serves `outputs/figures/confusion_matrix.png`. |
| `GET /api/per-class-accuracy` (+ `/api/chart/...`) | Per-class table as JSON or a bar chart. |
| `dataset_data()` | Cached 20-class nutrition; refreshes within 15 s are coalesced. |
| `_dataset_chart()` + 3 dataset chart routes | Calorie-vs-protein scatter, macro composition, calorie density. |
| `GET /api/chat/status` | Reports whether a Gemini key exists and which model is used. |
| `POST /api/chat` | Stateless chat turn; the client sends the meal context and the last 8 history turns. |
| `too_large()` | Handles 413 errors (32 MB upload cap). |

- **Imports from the project:** `predict` (`load_model`, `predict_pil`), `pipeline`
  (`run_pipeline`, `CLASSIFIER_CONF_THRESHOLD`), `charts` (12 names),
  `dataset_nutrition`, `evaluate` (`compute_confusion`, `per_class_report`),
  `gemini_chat` (`ask_gemini`, `read_api_key`, `GeminiError`, `MODEL_NAME`)
- **Imported by:** nothing (entry point)

#### `app.py` — ~809 lines
**Purpose:** The earlier Streamlit UI, kept as a fallback. It has two pages,
Food Analysis and Model Training, and no Dataset or Chat page.

| Function | What it does |
|---|---|
| `load_classifier()` | Loads the classifier once per session via `@st.cache_resource`; returns `(model, classes, device)`. |
| `_conf_class()` | Maps a confidence to a CSS badge class. |
| `food_card()` / `unrecognized_card()` | HTML cards for recognized and unrecognized regions. |
| `summary_metrics()` | The four total-metric tiles. |
| (module body) | Sidebar radio (line ~593), upload → `run_pipeline` (~664), training-run selectbox over `load_history` (~769), confusion-matrix image (~800). |

- **Imports from the project:** `predict.load_model`, `pipeline`
  (`run_pipeline`, `CLASSIFIER_CONF_THRESHOLD`), `charts` (palette constants,
  `annotate_image`, `make_macro_pie`, `make_calorie_bar`, `load_history`,
  `training_plot`)
- **Imported by:** nothing (entry point)

#### `test.py` — 1 line (`import pandas`)
Scratch file, not a test. Nothing imports it.

---

### `src/` — pipeline modules

#### `src/pipeline.py` — ~170 lines
**Purpose:** The single end-to-end entry point: detect → classify → apply the
threshold → merge duplicates → look up nutrition.

| Function | What it does |
|---|---|
| `run_pipeline(image_path, model, classes, device, det_conf=0.1, top_k=3)` | Returns `{regions, merged, unrecognized}`. |
| `main()` | CLI that prints a per-item and total nutrition table for one photo. |

- **Imports:** `detect.detect_food`, `nutrition.lookup_nutrition`, `predict` (`load_model`, `predict_pil`)
- **Imported by:** `server.py`, `app.py`

#### `src/detect.py` — ~316 lines
**Purpose:** Finds candidate food regions with YOLOv8 (Open Images V7), used
class-agnostically, with its own post-processing.

| Function | What it does |
|---|---|
| `iou(a, b)` | Intersection-over-union of two boxes. |
| `containment(inner, outer)` | Fraction of `inner` that lies inside `outer`. |
| `cross_class_nms(dets, iou_thresh=0.55)` | Keeps the highest-confidence box per overlapping region, ignoring labels. |
| `drop_container_boxes(dets, img_area, ...)` | Drops a big box (a plate or tray) that contains ≥2 item boxes. |
| `detect_food(image_path, ...)` | Runs YOLO, then the blocklist and degenerate-box filter, min-area filter, cross-class NMS and container suppression; attaches PIL crops. |
| `show_preview()` | OpenCV window with the boxes drawn (CLI only). |
| `main()` | CLI with `--conf --weights --iou --max-det --agnostic-nms --min-area --debug --no-preview`. |

- **Imports:** none from the project (`ultralytics`, `cv2`, `PIL`)
- **Imported by:** `pipeline.py`

#### `src/predict.py` — ~77 lines
**Purpose:** Loads a classifier checkpoint and runs top-k inference on an image.

| Function | What it does |
|---|---|
| `load_model(checkpoint_path, device)` | Builds MobileNetV3-Large with an N-class head from the checkpoint's own `classes`; returns `(model, classes)`. |
| `predict_pil(model, classes, pil_image, device, top_k=3)` | Eval transform → softmax → top-k `(label, prob)` list. |
| `predict(...)` | Same as `predict_pil`, but opens a file path first. |
| `main()` | CLI for a single image. |

- **Imports:** none from the project
- **Imported by:** `pipeline.py`, `evaluate.py`, `quick_test.py`, `server.py`, `app.py`

#### `src/nutrition.py` — ~326 lines
**Purpose:** Turns a class name into per-serving calories and macros from USDA
FoodData Central, with a hard-coded fallback table.

| Function | What it does |
|---|---|
| `_canonical_key()` | Normalizes a name (lowercase, underscores) and resolves aliases. |
| `_read_api_key()` | Reads `usda_api_key.txt`, or returns None. |
| `_extract_nutrients()` | Pulls per-100 g kcal, protein, carbs and fat from a USDA record; None if incomplete. |
| `_score_match()` | Ranks USDA descriptions: 0 for non-dish words, 5 → 1 by name closeness, −2 for restaurant-chain records (floor 1). |
| `_usda_query()` | Search call with a dataType filter, retried unfiltered on HTTP 400. |
| `lookup_nutrition(food_name)` | API → scale to serving size → otherwise fallback table → otherwise `source: "none"`. |
| `main()` | CLI lookup. |

- **Imports:** none from the project
- **Imported by:** `pipeline.py`, `dataset_nutrition.py`

#### `src/dataset_nutrition.py` — ~139 lines
**Purpose:** Builds and caches a nutrition table for all 20 classes (for the
Dataset Analysis tab), using the same `lookup_nutrition` as the photo pipeline.

| Function | What it does |
|---|---|
| `_per_100g()` | Scales a per-serving value back to per-100 g. |
| `_macro_percentages()` | Energy share by macro using 4/4/9 kcal per gram. |
| `_build_row()` | One class → a row of serving, per-100 g and macro-percentage figures. |
| `build()` | Looks up every class; returns a report dict. |
| `_read_cache()` / `load()` | Reads `outputs/dataset_nutrition.json`, or rebuilds it when missing, stale or refreshed. |
| `main()` | CLI table (`--refresh`). |

- **Imports:** `nutrition` (`SERVING_SIZES`, `lookup_nutrition`). `CLASS_NAMES = sorted(SERVING_SIZES)`
- **Imported by:** `server.py`

#### `src/charts.py` — ~438 lines
**Purpose:** All matplotlib figures and box drawing, shared by both front ends.
It returns Figures, PIL images or bytes and imports no UI framework.

| Function | What it does |
|---|---|
| `annotate_image()` | Draws coloured boxes and labels on the photo. |
| `_base_fig()` | Figure with the house style. |
| `make_macro_pie()` | Protein/carbs/fat donut. |
| `make_calorie_bar()` | Horizontal kcal-per-item bars (applies ×count). |
| `load_history()` | Splits `history.csv` into runs (skips `smoke` rows; a `head` row at epoch 1 starts a run) and labels them from `RUN_LABELS`. |
| `training_plot()` | Train and validation curves, with head and full phases shaded. |
| `make_per_class_accuracy_bar()` | Per-class accuracy bars, weakest class first. |
| `_place_labels()` | Collision-avoiding scatter labels. |
| `make_calorie_protein_scatter()` / `make_macro_composition_bar()` / `make_calorie_density_bar()` | The three dataset charts. |
| `fig_to_png_bytes()` / `pil_to_png_bytes()` / `to_base64()` | Serialization helpers. |

- **Imports:** none from the project
- **Imported by:** `server.py`, `app.py`

#### `src/gemini_chat.py` — ~214 lines
**Purpose:** Answers dietary questions with Gemini, grounded in the analysed
meal and bounded by a system instruction that refuses clinical or medication
questions.

| Function | What it does |
|---|---|
| `read_api_key()` | Reads `gemini_api_key.txt`, or returns None. |
| `format_meal_context()` | Turns the recognized items and totals into a text block for the prompt. |
| `_history_to_contents()` | Prior turns (each capped at 4,000 chars) plus the current question with MEAL CONTEXT. |
| `ask_gemini()` | Lazily imports `google.generativeai` and calls `MODEL_NAME = "gemini-3.5-flash-lite"` (temperature 0.4, 2048 max tokens, 45 s timeout); every failure becomes a `GeminiError`. |
| `_friendly_error()` | Maps SDK exceptions to user-facing messages. |
| `main()` | CLI with a stand-in macaroni-and-cheese meal. |

- **Imports:** none from the project
- **Imported by:** `server.py`

---

### `src/` — offline scripts

#### `src/train_classifier.py` — ~413 lines
**Purpose:** Fine-tunes ImageNet-pretrained MobileNetV3-Large on the 20-class
subset in two phases, with a leak-proof train/val/test split.

| Function / class | What it does |
|---|---|
| `RemappedSubset` | Dataset wrapper that remaps Food-101 labels to 0..19 and applies a transform. |
| `set_seed()` | Seeds `random`, NumPy, torch and CUDA; sets the cuDNN deterministic/benchmark flags. |
| `_seed_worker()` | Module-level DataLoader worker seeder (so Windows can pickle it). |
| `_image_key()` | `class/image-id` string for a dataset index. |
| `_fingerprint()` | SHA-1 (first 16 hex chars) over a split's image keys. |
| `make_split()` | Stratified random hold-out of `val_per_class` per class, using its own `random.Random(split_seed)`. |
| `load_split()` | Reuses `outputs/split_indices.json` if the config matches and the fingerprint verifies; otherwise builds and writes it. |
| `verify_split()` | Proves train, val and test are pairwise disjoint by image key; prints the counts. |
| `build_loaders()` | Train loader (augmented, shuffled, seeded generator) and val loader (eval transform). |
| `run_epoch()` | One pass under AMP autocast; backprop via GradScaler when an optimizer is given. |
| `log_row()` | Appends one CSV row to the history file. |
| `main()` | Argument parsing, split, model, and the two-phase loop with best-checkpoint saving. |

- **Imports:** none from the project. It reads `outputs/subset_meta.json`, written by `prepare_data.py`.
- **Imported by:** nothing

#### `src/prepare_data.py` — ~78 lines
**Purpose:** Downloads Food-101 and writes `outputs/subset_meta.json` (the
20-class list and the per-class cap).

| Function | What it does |
|---|---|
| `main()` | Downloads the train and test splits, validates the `CLASSES` names, counts images, writes the meta file. |

- **Imports / imported by:** none

#### `src/evaluate.py` — ~146 lines
**Purpose:** Scores the checkpoint on Food-101's official test split (5,000
images) and saves the confusion matrix.

| Function / class | What it does |
|---|---|
| `RemappedTest` | Test-split dataset wrapper with label remapping. |
| `build_test_loader()` | Loader over the test images of the checkpoint's classes. |
| `compute_confusion()` | Runs the model and returns an N×N confusion matrix. |
| `per_class_report()` | Per-class accuracy and most-confused class, worst first, plus overall accuracy. |
| `evaluate()` | Loads the checkpoint and returns `(classes, confusion)`. |
| `main()` | Prints the table and saves `outputs/figures/confusion_matrix.png`. |

- **Imports:** `predict.load_model`
- **Imported by:** `server.py` (`compute_confusion`, `per_class_report`)

#### `src/quick_test.py` — ~72 lines
**Purpose:** Spot check: classifies one random test-split photo per class
(`random.seed(42)`).

| Function | What it does |
|---|---|
| `load_test_split_by_class()` | Reads `data/food-101/meta/test.txt` → image paths per class. |
| `main()` | Predicts one image per class and prints right or wrong; the closing note cites 89.7%. |

- **Imports:** `predict` (`load_model`, `predict`)
- **Imported by:** nothing

#### `src/verify_setup.py` — ~33 lines
**Purpose:** Environment check: Python and PyTorch versions, a CUDA matmul
test, then import checks for `ultralytics`, `streamlit`, `matplotlib`,
`pandas`, `requests` and `PIL`. No functions; exits with code 1 if CUDA is
missing.

---

### Front end

#### `templates/index.html` — ~341 lines
Page structure only, as a Jinja template. It contains:
- The collapsible sidebar with three nav items (`data-view="analysis" | "training" | "dataset"`)
- Three `<section class="view">` panels: `view-analysis` (upload dropzone,
  results, charts, goal bar, warnings, Gemini chat form), `view-training`
  (run `<select>`, loss and accuracy `<img>`s, per-class chart, confusion
  matrix) and `view-dataset` (three chart images, captions, refresh button)
- The Inter font from Google Fonts, `style.css`, `window.DAILY_GOAL = {{ daily_goal }}`, and `app.js`

#### `static/app.js` — ~897 lines
All behaviour, as one ES5 IIFE with no framework or build step:
- **Navigation:** `showView()`, sidebar collapse
- **Upload:** file picker and drag-and-drop → `POST /api/analyze` → `render()`
- **Result rendering:** `foodCard`, `unrecognizedCard`, `metricCards`, `confBadge`, `densityLine`
- **Portion adjuster:** a slider and number box per item, from 25% to 300% of the standard serving
  (`portionControl`, `setPortion`, `applyPortions`). Everything is recomputed
  client-side from per-100 g figures, and chart requests are debounced by 250 ms (`scheduleCharts`).
- **Charts:** `POST /api/chart/calorie-bar` and `/api/chart/macro-pie` → blob images
- **Goal bar and warnings:** `renderGoal`, `renderWarnings` with `WARN_CARB_G = 60`,
  `WARN_FAT_G = 35` and `WARN_KCAL_SHARE = 0.4` of `DAILY_GOAL` (lines 340–342)
- **Chat:** `checkChatKey` (`/api/chat/status`), `sendChat` (`/api/chat`), a small Markdown renderer, `escapeHtml`
- **Dataset tab:** `loadDataset` (`/api/dataset-nutrition` plus three chart URLs), `datasetCaptions`
- **Training tab:** `loadTrainingRuns` (`/api/training-runs`), `showRun`, `loadPerClassAccuracy`

#### `static/style.css` — ~553 lines
All styling. It holds the palette as CSS variables (`--blue #2563EB`, `--navy`,
`--slate`, `--green`, `--amber`, `--red`, `--bg`, `--border`, lines 6–17),
which is duplicated in `charts.py`. It also contains
`[hidden] { display:none !important; }` (line 25) plus the sidebar, upload,
cards, charts, goal bar, chat, portion adjuster, warning flags, dataset tab and
a responsive breakpoint at 980 px.

---

## Part 2 — Exact current values

### `src/train_classifier.py`

**1. Optimiser, learning rate, weight decay, scheduler**

| Item | Value | Location |
|---|---|---|
| Optimiser | `torch.optim.AdamW`, created fresh at the start of each phase over parameters with `requires_grad=True` | `:365–366` |
| LR, head phase | `1e-3` | `:359` |
| LR, full phase | `1e-4` | `:360` |
| LR, smoke | `1e-3` (backbone frozen) | `:355` |
| Weight decay | `1e-4` (both phases) | `:366` |
| Scheduler | `CosineAnnealingLR(optimizer, T_max=epochs)`, recreated per phase and stepped once per epoch; no warm-up, `eta_min` defaults to 0 | `:367`, `:379` |

**2. Epochs and batch size**

| Item | Value | Location |
|---|---|---|
| Head epochs | `3` (`--head-epochs`) | `:290` |
| Full epochs | `7` (`--full-epochs`) | `:291` |
| Smoke | 1 epoch, 20 batches each for train and val | `:355`, `:374` |
| Batch size | `32` (`--batch-size`), same for train and val | `:288` |
| DataLoader workers | `4` (`--workers`), `pin_memory=True`, persistent workers | `:289`, `:230–241` |

**3. Loss function**
`nn.CrossEntropyLoss(label_smoothing=0.1)` at `:347`.

**4. Transforms** (`build_loaders`)

Training, `:208–214`:
1. `RandomResizedCrop(224, scale=(0.7, 1.0))`
2. `RandomHorizontalFlip()` (p = 0.5)
3. `ColorJitter(0.2, 0.2, 0.2)` (brightness, contrast, saturation; no hue)
4. `ToTensor()`
5. `Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])` (ImageNet)

Eval (validation), `:215–220`:
`Resize(256)` → `CenterCrop(224)` → `ToTensor()` → the same `Normalize`.

The same eval transform is defined again, identically, in `predict.py:16–21`
(used for inference and pipeline crops) and `evaluate.py:24–29`.

**5. Mixed precision (AMP)**
Yes. `GradScaler` and `autocast` are imported from `torch.amp`, falling back to
`torch.cuda.amp` (`:47–52`). The forward pass and loss run under
`autocast("cuda")` for training and validation (`:259–262`). The scaler is
created at `:348`, and backprop uses `scaler.scale(loss).backward()`,
`scaler.step`, `scaler.update` (`:266–268`).

**6. Best checkpoint selection**
- **Metric:** validation accuracy `va_acc` on the held-out validation set (`:377`).
- **Scope:** across both phases. `best = 0.0` is set once before the phase
  loop (`:352`) and never reset. A checkpoint is saved whenever
  `va_acc > best` (strictly greater) and it isn't a smoke run (`:391`).
- **Saved to:** `models/classifier_best.pt`, containing `state_dict`, `classes`,
  `val_acc`, `seed` and `split{split_seed, val_per_class, train_per_class, val_fingerprint}` (`:393–404`).
- **Current checkpoint** (read from the file): `val_acc 0.8495`, `seed 42`,
  `split_seed 1234`, `val_per_class 100`, `train_per_class 650`,
  `val_fingerprint 4500896ec52c0613`. In `history.csv`, run 4 peaks at full-phase
  epoch 7 (0.8495, the last epoch), so the final epoch was the one saved.

**7. Seeding**
- `--seed` defaults to `42` (`:298`); `set_seed(args.seed, args.deterministic)` is called at `:316`.
- `set_seed` (`:70–80`) calls `random.seed`, `np.random.seed`, `torch.manual_seed`
  and `torch.cuda.manual_seed_all`, sets `cudnn.deterministic = deterministic`
  (default **False**) and `cudnn.benchmark = not deterministic` (default **True**).
- The train DataLoader gets a `torch.Generator` seeded with `seed` (`:226–227`,
  `:234`), which controls shuffle order and the base seed of worker torch RNGs.
- `worker_init_fn = partial(_seed_worker, base_seed=seed)` re-seeds `random` and
  `numpy` per worker with `seed + worker_id` (`:83–86`, `:228`). It is applied
  to both loaders; only the train loader gets the generator.
- The split has its own seed: `--split-seed` defaults to `1234` (`:44`, `:300`)
  and is used only through `random.Random(seed)` inside `make_split` (`:110`).

**8. Train/validation split**
- The pool is Food-101 `split="train"` only (750 per class) (`:126`).
- `make_split` (`:101–121`): per class, sort the indices, shuffle with
  `Random(1234)`, take the first **100** as validation (then sorted) and the
  remaining **650** as training (left in shuffled order so a cap takes a random
  subset). It is stratified: exactly 100 per class.
- **Stored in** `outputs/split_indices.json` (`SPLIT_PATH`, `:38`), with
  `classes`, `split_seed`, `val_per_class`, `train_per_class`, `val_fingerprint`,
  `train_fingerprint`, `train_indices` and `val_indices` per class. The current
  file was created 2026-09-16T22:57:28: 13,000 train and 2,000 val indices,
  fingerprints val `4500896ec52c0613` and train `1700e94314e67e66`.
- **Reuse rules** (`load_split`, `:124–168`): rebuilt if classes,
  `val_per_class` or `split_seed` differ, or with `--resplit`. If the recomputed
  validation fingerprint doesn't match, the run aborts.
- **Cap** (`:328–331`): without `--full-data`, training is thinned to
  `max_train_per_class` from `outputs/subset_meta.json` (currently **200**).
  Validation is never capped.
- **Class list** comes from `outputs/subset_meta.json["classes"]` (`:317–318`),
  which `prepare_data.py` writes. It currently matches the split file.
- `verify_split` (`:171–203`) runs every time and aborts if any image key is
  shared between train/val, train/test or val/test.

### `src/detect.py`

**9. YOLO weights and inference parameters** (`detect_food` signature, `:109–111`; YOLO call, `:113–116`)

| Parameter | Value |
|---|---|
| Weights | `yolov8m-oiv7.pt` (Open Images V7, medium) |
| Confidence threshold | `0.1` (the pipeline passes `det_conf=0.1`, `pipeline.py:30,36`) |
| YOLO internal NMS IoU | `0.45` |
| `max_det` | `50` |
| `agnostic_nms` | `False` |
| `imgsz` | not set, so the Ultralytics default (640) applies |
| Min box area | `min_area_frac=0.02` (2% of image area, `:166–168`) |
| Degenerate boxes | dropped if `x2<=x1` or `y2<=y1` (`:146`) |
| Model loading | `YOLO(weights)` is constructed on every call (`:113`), i.e. once per analysed photo |

The pipeline overrides only the confidence threshold; everything else uses
these defaults. The CLI defaults are the same (`:273–292`).

**10. NMS and container suppression**

| Item | Value | Location |
|---|---|---|
| Cross-class NMS IoU | `0.55` | `:77` (`cross_class_nms`) |
| Container area threshold | `cover_thresh=0.55` (box ≥55% of the image area) | `:87`, `:97` |
| Minimum children | `min_children=2` | `:87`, `:104` |
| Child containment | `> 0.8` of the child inside the container (hard-coded) | `:102` |
| Early exit | skipped when fewer than 3 boxes remain (`len < min_children + 1`) | `:90` |
| Safety net | `return keep or dets` (never returns empty) | `:106` |

Order of steps: YOLO (+ its NMS) → blocklist and degenerate filter → min-area →
cross-class NMS → container suppression → sort by confidence → crop (`:139–235`).

**11. `NON_FOOD` blocklist**
**126 entries** (`:20–53`), verified by importing the set: people, furniture,
cutlery, drinks, appliances, room objects, misc. Plates and bowls are
deliberately not blocked.

### `src/pipeline.py`

**12. Confidence threshold**
- `CLASSIFIER_CONF_THRESHOLD = 0.40` at `pipeline.py:27`.
- Applied at `pipeline.py:45`: `if top_conf < CLASSIFIER_CONF_THRESHOLD` → the
  region goes to `unrecognized` with `region_idx`, `conf` and `top_guess`.
  Exactly 0.40 counts as recognized.
- It is also imported by `server.py:28`, where it decides the box label
  (name or `"?"`) at `server.py:194`, and by `app.py:22`. `server.py:149` passes
  it to the template, but `index.html` never references `conf_threshold`.

**13. Duplicate merging** (`pipeline.py:58–70`)
- The key is the classifier's top-1 label string, not box overlap.
- The first occurrence creates `{name, conf, count: 1, alts}`. Each later region
  with the same label increments `count` and sets `conf` to the maximum
  confidence seen. `alts` is kept from the first occurrence.
- Output order is first appearance; regions arrive sorted by detection confidence (`detect.py:232`).
- Nutrition is looked up once per merged entry (`:72–73`) and stays per
  serving. Multiplication by `count` happens in the consumers: `server.py:205–208`,
  `pipeline.main` `:128–131` and `charts.make_calorie_bar`.
- Unrecognized regions are never merged.

### `src/nutrition.py`

**14. USDA dataTypes and query overrides**
- The first request asks for `dataType = "SR Legacy"` and `"Survey (FNDDS)"`,
  with `pageSize 10` and an 8 s timeout (`:191–197`, `:206`).
- On HTTP 400 only, it retries without a dataType filter, which can return any
  type, including Branded (`:198–211`). Ranking then relies on `_score_match`.
- **Query overrides: 10 classes** (`_QUERY_OVERRIDES`, `:230–241`):
  pizza → "pizza cheese", donuts → "doughnut, NFS", french_fries → "potato,
  french fries, NFS", pancakes → "pancakes, plain", waffles → "waffle, NFS",
  cheesecake → "cheesecake, plain", tacos → "taco, beef, NFS", onion_rings →
  "onion rings, NFS", paella → "paella seafood rice", omelette → "egg omelet plain".
- Also: `_PREFER_FALLBACK = {"caesar_salad"}` (`:244`) skips the API entirely for
  that class; 22 aliases (`:75–98`); a 20-entry `SERVING_SIZES` (`:26–47`) and
  `_FALLBACK` (`:51–72`); a 150 g default for unknown names (`:223`).
- The current cache (`outputs/dataset_nutrition.json`, generated
  2026-09-16T21:14Z) has 19 classes from the API and 1 from the fallback (caesar_salad).

### General

**15. Total Python lines**
**3,683 lines** (`wc -l`) across 15 files: `app.py` 809, `server.py` 452,
`train_classifier.py` 413, `charts.py` 438, `nutrition.py` 326, `detect.py` 316,
`gemini_chat.py` 214, `pipeline.py` 170, `evaluate.py` 146,
`dataset_nutrition.py` 139, `prepare_data.py` 78, `predict.py` 77,
`quick_test.py` 72, `verify_setup.py` 33, `test.py` 0. That is **2,422** in `src/`
and **1,261** at the root. The front-end files add 1,791 more (HTML 341, JS 897, CSS 553).

**16. Does anything use the test split during training or checkpoint selection?**
**No.**
- Every training and validation image comes from `Food101(split="train")`
  (`train_classifier.py:126`), and both loaders are built from that dataset only (`:230–241`).
- The checkpoint is chosen on `va_acc` from the validation loader (`:377`, `:391`).
- The only test-split access in `train_classifier.py` is in `verify_split`
  (`:173–180`). It reads the test split's file names and labels (`_image_files`,
  `_labels`) to prove no image overlaps. It never decodes an image, builds a
  loader or feeds anything to the model.
- Test-split readers outside training: `evaluate.py:48` (the final score),
  `quick_test.py:21` (spot check), `server.py:131` (via `compute_confusion`,
  cached for the per-class chart), and `prepare_data.py:42` (download and count only).
- The historical exception is runs 1–3 in `history.csv` and the archived
  `models/old_split_run/`, which used the test split as validation. The
  current checkpoint doesn't.
- Consistency check: `classifier_best.pt` modified 23:04:03,
  `confusion_matrix.png` 23:04:54, `per_class_accuracy.json` 23:12:41 (all
  2026-09-16), so the test figures postdate the checkpoint. The cache reports
  0.8972 over 5,000 images.

---

## Discrepancies: documentation vs. code

Ordered by how likely each is to come up in a defense.

| # | Documentation says | Code / data says | Where |
|---|---|---|---|
| 1 | `Reference.MD:66`: "20 classes, full training data: **90.7%** (current, final model)" | The current model scores **89.7% test / 85.0% val** on 650 per class. 90.7% was run 3, whose validation set was the test split. | `charts.py` `RUN_LABELS`; checkpoint `val_acc 0.8495`; `per_class_accuracy.json` 0.8972 |
| 2 | `Reference.MD:70–76`: "~3,750 test photos"; weakest is **filet_mignon 62%**, confused with steak; others 86–98% | Neither class is in the current 20. The evaluation covers **5,000** images. Weakest now: **falafel 80.8%** (→ tacos ×11), **chicken_curry 82.4%** (→ chicken_wings ×13), omelette 85.2%. Best: pizza 96.4%. The range is 80.8–96.4%. | `outputs/per_class_accuracy.json` |
| 3 | `Reference.MD:222`: "the classifier's **9%** error rate" | Test error is **10.3%** (100 − 89.7). | as above |
| 4 | `Reference.MD:35` "only a small new final layer learns"; `train_classifier.py:6` "trains only the new head"; `Reference.MD:44` calls it "linear probing" | Only `model.features` is frozen (`:363–364`). The whole `classifier` block trains in phase 1: the **pretrained** `Linear(960→1280)` plus the new `Linear(1280→20)`, **1,255,700 of 4,227,652** parameters. Because `model.train(True)` is set (`:248`), BatchNorm running statistics in the "frozen" backbone still update during phase 1. | `train_classifier.py:248, 344, 363–365` |
| 5 | `Reference.MD:272`: "Most protein: chicken wings **22.6 g/100g**" | The current cache has chicken_wings at **17.0 g/100g** (USDA match "CHICKEN WINGS"). The other quoted figures still match: falafel 514, donuts 426, cheesecake 399, sushi 94, curry 107, mac & cheese 110, fries 2.5 g. | `outputs/dataset_nutrition.json` |
| 6 | `CLAUDE.md`: USDA results are "scored to **reject** … branded-chain hits"; `Reference.MD:139–142`: fixed by "targeting NFS entries" | Chain records are **penalised** (score −2, floor 1), not rejected (`nutrition.py:163, 181`). Only 5 of the 10 overrides use "NFS". If the filtered query 400s, the unfiltered retry can include Branded data. | `nutrition.py:148–211, 230–241` |
| 7 | `history.csv` `lr` column reads as "LR used that epoch" | It's logged **after** `sched.step()` (`:379–382`), so it shows the next epoch's LR. Head epoch 1 shows 0.00075 but trained at 1e-3, and the last epoch of each phase shows 0.0. | `train_classifier.py:379–382` |
| 8 | `train_classifier.py:3`: running with no flags is the "full two-phase run"; `CLAUDE.md`: `MAX_TRAIN_PER_CLASS` is "bypassed by `--full-data`" | Without `--full-data`, training is capped at **200 per class**. The cap is read from `outputs/subset_meta.json`, not from the constant, so editing `prepare_data.MAX_TRAIN_PER_CLASS` has no effect until `prepare_data.py` is re-run. | `train_classifier.py:317, 328` |
| 9 | `CLAUDE.md`: `server.py` passes the 0.40 threshold "into the template" | It is passed (`server.py:149`), but `index.html` never uses `conf_threshold`. | `templates/index.html` |
| 10 | `prepare_data.py` prints "Capped for training: ~4000 train / ~1600 test" | `MAX_TEST_PER_CLASS = 80` is not used anywhere. Evaluation always uses 250 per class, and the train figure ignores the 100-per-class validation hold-out. | `prepare_data.py:32–33, 67–71` |
| 11 | `CLAUDE.md`: the classifier is loaded once | True for the classifier, but YOLO weights are reloaded on every `detect_food` call (`detect.py:113`). `server.py:190–195` also re-classifies every region a second time just to label the boxes, repeating inference `run_pipeline` already did. | `detect.py:113`, `server.py:190–195` |
| 12 | `CLAUDE.md`: `--smoke` never lands in the real history | True now, but `history.csv` still holds two legacy `smoke` rows (before runs 1 and 3). `charts.load_history` skips them, so the graphs are unaffected. | `outputs/history.csv`, `charts.py:141–145` |
| 13 | `verify_setup.py` is the "package sanity check" | It checks `requests` (unused, since `nutrition.py` uses `urllib`) and `streamlit`, but not `flask` or `google-generativeai`, and it exits with an error on a machine without CUDA. | `verify_setup.py:21, 26` |

Checked and **consistent**: the 0.40 threshold; the 100/650/250 split sizes;
seed 42 and split seed 1234; `RUN_LABELS` against `history.csv` (84.7 / 89.5 /
90.7 / 85.0 best val); `quick_test.py`'s 89.7% note; the warning thresholds
60 g carbs, 35 g fat and 800 kcal (0.4 × 2000) in `app.js:340–342`; the
25%–300% portion range and 250 ms chart debounce; three tabs in the Flask UI;
`SERVING_SIZES` and `_FALLBACK` covering exactly the 20 classes; roughly 5.5
minutes total for run 4's epochs (the docs say "~6 min").
