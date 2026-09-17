# FoodLens — Defense Prep (Plain Language)

Read this the night before, out loud if possible. Everything here is
already true and already tested — this is just the simple version.

## What the project does

You upload a photo of a plate. The system finds each separate food
item, names it, and tells you calories and nutrition for it — all
automatically, no typing required.

## The two "brains"

**Brain 1 (YOLO) finds food.** It draws a box around anything that
looks like food. It does NOT know food names specifically — it just
knows "something is here."

**Brain 2 (MobileNetV3) names food.** For each box Brain 1 draws,
Brain 2 looks at just that piece and says exactly what dish it is,
from a list of 20 foods it was trained on.

**Why two brains, not one:** finding things and precisely naming
things are different skills. A general detector is good at "where,"
not "exactly what." Splitting the job this way is a standard,
well-established pattern (used in retail shelf scanning, wildlife
camera traps, etc.), not something improvised for this project.

## How Brain 2 was actually taught

It didn't start from nothing. It already knew how to "see" — edges,
shapes, textures — from being shown millions of general photos before
this project ever touched it. Teaching it your 20 foods happened in
two steps:

1. **Head phase:** only a small new final layer learns; everything
   else is frozen. Safe to move fast here, since nothing valuable is
   at risk yet.
2. **Full phase:** everything unfreezes, but learning is much slower
   and gentler. Now the valuable pretrained knowledge IS at risk, so
   big adjustments could damage it — small steps protect it while
   still adapting it to food.

This is called **transfer learning** / **fine-tuning**, and the
two-step approach is called **linear probing then full fine-tuning**
— a standard technique, not invented for this project.

## Why 20 classes, not all 101 in Food-101

Started at 15, then expanded to 20 — a deliberate scope decision made
to have a fully working, well-tested pipeline within the timeline.
Chosen to include: foods that are globally popular, a mix of
vegetarian and chicken dishes, and two genuinely Indian dishes
(chicken_curry, samosa) — the only two Food-101 offers, since the
dataset is heavily Western/East-Asian biased overall.

**If asked "why not more classes":** "I prioritized a complete,
thoroughly tested pipeline over a wider but less-tested one, given
the timeline. More classes is a natural next step, and the code
already supports it — it just needs retraining."

## Real results

- 15 classes, capped training data (200 imgs/class): **84.7%**
- 15 classes, full training data (750 imgs/class): **89.5%**
  (proves more data helps — same setup, only data volume changed)
- 20 classes, full training data: **90.7%** (current, final model)

## Checking where it's weak (confusion matrix)

Built a full evaluation across ~3,750 real test photos, not just a
summary number. Found: **filet_mignon** is the weakest class (62%),
almost entirely confused with **steak** (a real cut of steak vs. a
dish literally made from a specific steak cut — genuinely hard to
tell apart, even for people). This pair was deliberately chosen to
stress-test the model on purpose — the confusion matrix confirmed
that instinct was correct. Nearly every other class scores 86-98%.

## The "I don't know" safety net (confidence threshold)

The classifier ALWAYS outputs a guess — it has no built-in "I don't
know" option. So a rule was added: if its top confidence is below
**40%**, the system shows "Unrecognized food" instead of the guess,
rather than confidently stating something wrong.

**Real proof this works:** tested on an Indian thali (curry, chapati,
onions — chapati/onion aren't in the 20 classes). The curry was
correctly identified at 85%. Chapati got guessed internally as
"tacos" at 36% and correctly rejected — shown as unrecognized instead
of a wrong confident answer.

**The honest cost:** this occasionally rejects a real, correct item
too — a donut in a group photo was correctly guessed at only 28% and
got rejected, undercounting that photo by one donut. This is a
deliberate trade-off: prioritizing "never confidently wrong" over
"never misses anything."

## The detection stage in more detail

**YOLO is used exactly as downloaded — never trained or fine-tuned.**
This was a deliberate time-scope decision: annotating a custom food
dataset for detection would have taken real time for a piece that
isn't the thesis's core contribution.

**Why Open Images instead of COCO:** COCO only matched 3 of the 20
food classes exactly. Open Images V7 (601 categories) matched 9 —
still incomplete, but meaningfully better, for the same zero-training
cost.

**Why "class-agnostic" detection:** general datasets don't have
category names for most specific dishes (no "chapati," no "ramen").
So instead of only keeping boxes with a food-specific label, the
system accepts almost ANY detected box except an explicit blocklist
of obviously non-food things (hands, forks, tables, cups...).
Whatever survives becomes a candidate region for Brain 2 to name.
This is what makes multi-item detection actually work.

**Known limitation, found through real testing:** items physically
touching/overlapping with no visible gap (e.g. fries piled against
onion rings) get detected as one region, not two — a real,
documented constraint of general-purpose detectors, not a bug.

## Combining duplicate detections

If two boxes both get named the same food (e.g. two chapati regions),
they're merged into one entry ("chapati x2") AFTER classification —
using the actual predicted name, not guessing from box overlap.
Two overlapping boxes don't necessarily mean the same food (a burger
and fries can touch); two boxes classified as the same name almost
certainly do.

## Turning a food name into nutrition facts

The food name is sent to **USDA FoodData Central**, a real government
nutrition database, the same as typing a search query. Real bugs
found and fixed during testing:

- Class names had underscores (`french_fries`) which the API didn't
  understand — fixed by converting to spaces before searching.
- Generic searches sometimes matched branded restaurant items (e.g.
  "tacos" matching "Taco Bell") instead of a generic entry — fixed by
  targeting USDA's "NFS" (Not Further Specified) generic entries
  specifically for problem classes.

**Fallback:** every class also has a local backup nutrition table,
used automatically if the live API fails for any reason — so a flaky
connection during a demo can't break the whole app.

## Connecting everything (the pipeline)

One file (`pipeline.py`) chains all of this: detect regions ->
classify each one -> check confidence -> merge duplicates -> look up
nutrition -> total everything -> return one complete answer. One
photo in, one full result out, no manual steps.

## The web app — two versions, and why

**First version: Streamlit.** Fast to build, worked, but its built-in
UI parts (sidebar, header) aren't fully controllable through custom
styling — specifically, positioning a custom element relative to
Streamlit's own built-in controls proved unreliable, since those
controls live in framework-generated code outside the app's own
structure. A known, documented Streamlit limitation.

**Second version: Flask + hand-built HTML/CSS/JS.** Switched so every
element is explicitly placed by hand — nothing hidden, nothing
generated automatically. This directly solved the positioning
problem: things that took many attempts to work around in Streamlit
were correct on the first try in the new version. All the actual AI
logic (detection, classification, nutrition, pipeline) was reused
completely unchanged — this was purely a presentation-layer decision.

The app has three tabs: **Food Analysis** (upload a photo), **Model
Training** (the training and evaluation graphs), and **Dataset
Analysis** (nutrition across all 20 classes, no photo needed).

## Gemini chat

Answers questions grounded in the specific meal just analyzed (its
foods, calories, macros) — e.g. "is this diabetes-friendly," "how do
I make this lower-calorie." Deliberately scoped to general nutrition
suggestions only, not medical advice, with a visible disclaimer.
Free-tier API has a request-rate limit, handled gracefully rather
than silently failing.

## The warning panel (high carb / high fat / high calorie)

Under the results there is a small card that comments on the meal as a
whole. Three simple rules: more than **60g of carbs**, more than **35g
of fat**, or more than **800 calories** in one meal each raise a flag.
If none of them do, it says the meal stays within all three — it never
just goes blank, because a blank panel looks like something broke.

Where the numbers come from: 60g is about a third of a typical daily
carbohydrate intake, 35g is about half a typical day's fat, and 800 is
40% of the 2000-calorie daily goal the app already uses.

**Tested both ways:** the samosa photo (1116 kcal, 119g carbs, 63g fat)
raises all three. The macaroni and cheese photo (446 kcal, 36g carbs)
raises none and shows the balanced message.

**The framing matters more than the rules.** Each line is a plain
statement of fact — "High carbohydrate content — 119g in this meal" —
with the reference value underneath. It never says "too much", never
says "unhealthy", never tells the person to do anything. There is a
permanent caption saying these are general reference values, not
medical or dietary advice, and that portions are estimates.

**If asked "isn't that basically medical advice?":** no — it reports a
comparison against a published general reference value and stops. It
doesn't interpret it for a particular person, doesn't diagnose, and
doesn't recommend. That boundary is the same one the chat draws.

## The portion slider (the most honest feature in the app)

**Admit this one first, because it's the real limitation:** the system
does NOT measure how much food is on the plate. It recognises the dish
and then assumes a standard weight for it — 200g of macaroni and
cheese, 170g of tacos. A tiny samosa and a huge samosa give the exact
same calorie number.

**This is the biggest source of error in the whole project — bigger
than the classifier's 9% error rate.** Getting the dish right but the
portion twice too big is a 100% error in the calories.

**So rather than hide it, it's exposed as a control.** Every food card
now has a slider and a grams box. Move it and that card's calories,
protein, carbs and fat recalculate instantly, and so does everything
else on the page: the four totals at the top, both charts, the daily
goal bar, the warning flags, and what the chat knows about your meal.
Nothing is sent to the server to do this — all four values scale in a
straight line from the per-100g figures already on the page. There's a
Reset link to go back to the standard serving.

For a merged item ("donuts x8") the slider sets the weight of ONE
donut and the x8 is applied on top, so correcting the size doesn't
accidentally change the count.

**If asked why the charts don't lag:** the two charts are drawn by the
server, so dragging a slider could fire a request per pixel. It waits
250ms after you stop moving and sends one. Measured: a 26-step drag
produced 1 request. The charts were deliberately kept server-drawn
rather than redrawn in the browser, because that same drawing code
produces the figures in the thesis — a second copy in JavaScript would
eventually drift and disagree with the write-up.

**The one-line version:** "the portion is an estimate, so I made the
estimate adjustable instead of pretending it was a measurement."

## The Dataset Analysis screen (third tab)

A third tab that has nothing to do with any uploaded photo. The first
tab asks "what's on this plate", the second asks "how good is the
model", this one asks **"what do these 20 foods actually look like
nutritionally?"** It works before you've uploaded anything.

**Three charts:**

1. **Calories vs protein** — one dot per food, calories across,
   protein up, each dot labelled. The clusters are the point: the
   desserts and fried things sit far right and low (lots of calories,
   almost no protein), the meat dishes sit high up, and sushi, curry
   and macaroni and cheese sit bottom-left, light in both.
2. **Macro composition** — all 20 foods, each bar split into protein /
   carbs / fat **as percentages of that dish's calories, not grams**.
   That's the trick: you can't compare a 60g samosa and a 250g curry
   by weight, but you can compare what they're made of.
3. **Calorie density ranking** — calories per 100g, highest to lowest,
   with a line at the average.

**Numbers worth knowing:** densest are falafel 514, donuts 426,
cheesecake 399 kcal/100g; lightest are sushi 94, chicken curry 107,
macaroni and cheese 110. Most protein: chicken wings 22.6g/100g. Least:
french fries 2.5g/100g.

**If asked "why is falafel above the desserts?":** because that's the
USDA entry for fried falafel, which genuinely is very dense. The app
doesn't curate or adjust the government database's numbers.

**Where the data comes from:** the exact same nutrition function the
photo pipeline uses. There is no second nutrition table for this page,
so this screen and a food card physically cannot disagree.

**Why it's cached:** 20 foods means 20 requests to USDA, which takes
**49 seconds**. Doing that every time someone opens the tab would be
unusable and would burn through the API limit. So it's fetched once,
saved to a file, and served from there in about 3 milliseconds. There's
a Refresh button to re-fetch on demand.

## "What about mAP?" (a likely question)

mAP measures detection accuracy against hand-labeled ground-truth
boxes. It requires a labeled detection dataset. Since YOLO here is
used purely pretrained (not trained/fine-tuned on a custom labeled
dataset), there's no ground truth to measure it against. The
classification stage DOES have real ground truth (Food-101's labels),
which is why accuracy and the confusion matrix were used there
instead.

## If you don't know something in the moment

It's fine to say: "That's a good question — let me think through
it," and reason out loud using what you actually understand about
the pipeline, rather than guessing at a number. Understanding the
*shape* of how each piece works matters more than memorizing exact
figures.