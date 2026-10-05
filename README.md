# FoodLens

FoodLens is a web app that estimates the nutrition of a meal from a photo. You
upload a picture of a plate; the app finds each food item on it, identifies what
it is, looks up its calories and macronutrients, and shows the totals for the
whole meal. It was built as a thesis project.

## Features

- **Multi-item detection** — several foods on one plate are found and analysed
  separately.
- **20 food classes** — from pizza and sushi to samosa and paella.
- **Nutrition from USDA FoodData Central** — calories, protein, carbohydrates
  and fat for a standard serving of each item.
- **Portion and quantity adjustment** — change the grams of each item, or the
  number of pieces for countable foods (sushi, donuts, chicken wings …); every
  total, chart and warning updates immediately.
- **Charts and nutrition warnings** — macronutrient split, calories per item,
  progress against a 2,000 kcal daily goal, and flags for high carbohydrate, fat
  or calorie content.
- **Nutrition assistant** — questions about the analysed meal, answered by
  Google Gemini and limited to general dietary guidance (login only).
- **User accounts** — sign up and log in with email and password (Supabase Auth).
- **Saving meals** — logged-in users can save an analysed meal to their account.
- **Guest mode** — try the app without an account: 3 free photo analyses.
- **Dataset Analysis and Model Training tabs** — nutrition across all 20
  classes, the training curves of every run, per-class accuracy and the
  confusion matrix.

## How it works

```
photo ──► YOLOv8 ──► food regions ──► MobileNetV3 ──► label + confidence
                                                          │
                              confidence < 0.40 ──► "unrecognised"
                                                          │
                       USDA FoodData Central ◄────────────┘
                                │
                  merge duplicates, total the meal ──► results page
```

1. **Find the food.** A YOLOv8 object detector (Open Images V7 weights) proposes
   regions in the photo. It is used only to find *where* things are: non-food
   detections (hands, cutlery, furniture, drinks) are discarded, overlapping
   boxes are reduced to one per region, and a box covering the whole plate is
   dropped when it contains several separate items.
2. **Name each region.** Each region is cropped and classified by a MobileNetV3
   model fine-tuned on 20 food classes.
3. **Admit uncertainty.** If the classifier's confidence is below 0.40, the
   region is reported as unrecognised rather than given a likely-wrong name.
4. **Look up nutrition.** Each recognised food is matched to a USDA FoodData
   Central record; values per 100 g are scaled to a standard serving size. One
   class (caesar salad) uses a built-in table because the USDA entry omits the
   dressing.
5. **Merge and total.** Repeated detections of the same food become one line
   item, and the meal totals are calculated.

## Results

The classifier reaches **89.9% accuracy on 5,000 held-out test images** (20
classes, 250 per class). These are Food-101's official test images: they were
never used for training or for choosing the model.

Training data is split three ways, per class: 650 images for training, 100 for
validation (used to pick the best checkpoint) and 250 for the final test. The
best validation accuracy was 85.1%. Validation reads lower than test because
Food-101's training images were deliberately left uncleaned by the dataset's
authors, while the test images were checked by hand.

Training uses two-phase transfer learning from ImageNet weights: first only the
new classification layer is trained (3 epochs, learning rate 1e-3), then the
whole network is fine-tuned at a lower learning rate (7 epochs, 1e-4).

## Tech stack

- Python 3.11, PyTorch, torchvision
- YOLOv8 (Ultralytics) for detection, MobileNetV3-Large for classification
- Flask for the web server
- Supabase (PostgreSQL and Auth) for accounts and saved meals
- USDA FoodData Central API for nutrition data
- Google Gemini API for the nutrition assistant
- Plain HTML, CSS and JavaScript for the front end (no framework, no build step)

## Project structure

```
server.py              Flask web app: pages, API endpoints, charts
auth.py                Login, sessions, guest mode, CSRF and access checks
meals.py               Saving meals: validation, nutrition recalculation, Supabase call
app.py                 Earlier Streamlit interface, kept as a fallback (no login)
src/
  pipeline.py          Detect → classify → nutrition, the core of the app
  detect.py            YOLOv8 detection and box post-processing
  predict.py           Loads the classifier and predicts one image or crop
  nutrition.py         USDA FoodData Central lookup, serving sizes, fallback table
  dataset_nutrition.py Nutrition for all 20 classes, cached for the Dataset tab
  charts.py            All charts (matplotlib), shared by both interfaces
  gemini_chat.py       Gemini nutrition assistant
  settings.py          Reads configuration from src/.env
  prepare_data.py      Downloads Food-101 and selects the 20 classes
  train_classifier.py  Trains the classifier (train/validation/test split)
  evaluate.py          Test accuracy, per-class report, confusion matrix
  quick_test.py        One test photo per class, for a quick visual check
templates/, static/    Web pages, styles and front-end JavaScript
supabase/schema.sql    Database tables, security policies and the save_meal function
outputs/               Training history, the data split, thesis figures
```

## Setup

Requires Python 3.11. A CUDA-capable GPU is needed for training; the app itself
also runs on a CPU, more slowly.

1. **Create the environment and install dependencies** (PyTorch is pinned to a
   CUDA 12.1 build, hence the extra index):

   ```
   py -3.11 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
   ```

2. **Create `src/.env`** with these variables, one `NAME=value` per line. The
   file is ignored by git; never commit real values.

   | Variable | Where to get it |
   |---|---|
   | `SUPABASE_URL` | Supabase dashboard → Project Settings (the project URL) |
   | `SUPABASE_PUBLISHABLE_KEY` | Supabase dashboard → Project Settings → API Keys — the *publishable* key, not the secret key |
   | `SECRET_KEY` | Any long random string, e.g. `python -c "import secrets; print(secrets.token_hex(32))"` |
   | `GEMINI_API_KEY` | Google AI Studio: https://aistudio.google.com/app/apikey |
   | `USDA_API_KEY` | https://fdc.nal.usda.gov/api-key-signup.html (free) |

   Without the USDA key, nutrition falls back to a built-in table; without the
   Gemini key, the assistant reports that it is unavailable.

3. **Set up the database** (new Supabase project only): run
   `supabase/schema.sql` in the Supabase SQL editor.

4. **Download the data, train and evaluate:**

   ```
   .venv\Scripts\python.exe src\prepare_data.py
   .venv\Scripts\python.exe src\train_classifier.py --full-data --seed 42
   .venv\Scripts\python.exe src\evaluate.py
   ```

   `prepare_data.py` downloads Food-101 (about 5 GB). Training takes a few
   minutes on a laptop GPU.

5. **Run the app:**

   ```
   .venv\Scripts\python.exe server.py
   ```

   Then open http://127.0.0.1:5000. The YOLOv8 weights download automatically
   on first start.

## Security

- **Sessions in httpOnly cookies.** Login tokens are stored in cookies that page
  scripts cannot read, and are verified on the server against Supabase's public
  signing keys.
- **Row-level security.** Database rules ensure each user can only read and
  write their own meals, even if a request bypasses the app.
- **CSRF protection.** SameSite cookies, an Origin check on every request that
  changes data, and JSON-only login endpoints.
- **Server-side nutrition.** When a meal is saved, the server recalculates all
  nutrition values itself from a signed copy of the analysis; values sent by
  the browser are not trusted.

## Limitations

- YOLOv8 is used off the shelf and is not trained on food, so items that touch
  or overlap can be merged into one region or missed.
- Portion sizes start from a standard serving per food; the user adjusts them
  by hand.
- The classifier knows only 20 classes. Anything else is either reported as
  unrecognised or confused with the closest known class.
- The model was evaluated on a single train/validation/test split, without
  cross-validation.

## Future work

- Meal history: browsing, editing and deleting saved meals.
- Deployment to a public server.
- Learning from user corrections to improve the classifier.
