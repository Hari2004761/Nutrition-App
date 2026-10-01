/* FoodLens front end — talks to the Flask API in server.py */
(function () {
  "use strict";

  var DAILY_GOAL = window.DAILY_GOAL || 2000;
  var $ = function (id) { return document.getElementById(id); };

  /* ══════════════════ SIDEBAR ══════════════════ */
  var sidebar = $("sidebar");

  $("collapseBtn").addEventListener("click", function () {
    sidebar.classList.toggle("collapsed");
    var collapsed = sidebar.classList.contains("collapsed");
    this.title = collapsed ? "Expand sidebar" : "Collapse sidebar";
    this.setAttribute("aria-label", this.title);
  });

  var TITLES = {
    analysis: "Culinary Intelligence",
    training: "Model Training Graphs",
    dataset:  "Dataset Analysis"
  };

  function showView(name) {
    document.querySelectorAll(".sb-nav-item").forEach(function (li) {
      li.classList.toggle("active", li.dataset.view === name);
    });
    $("view-analysis").hidden = name !== "analysis";
    $("view-training").hidden = name !== "training";
    $("view-dataset").hidden  = name !== "dataset";
    $("pageTitle").textContent = TITLES[name];
    if (name === "training") { loadTrainingRuns(); loadPerClassAccuracy(); }
    if (name === "dataset") { loadDataset(); }
  }

  document.querySelectorAll(".sb-nav-item").forEach(function (li) {
    li.addEventListener("click", function () { showView(li.dataset.view); });
    li.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); showView(li.dataset.view); }
    });
  });

  /* ══════════════════ FILE SELECTION ══════════════════ */
  var fileInput = $("fileInput");
  var dropzone = $("dropzone");
  var analyzeBtn = $("analyzeBtn");
  var selectedFile = null;

  function setFile(file) {
    if (!file) return;
    if (!/^image\/(png|jpe?g)$/.test(file.type)) {
      showError("Please choose a JPG, JPEG or PNG image.");
      return;
    }
    hideError();
    selectedFile = file;
    $("filenameText").textContent = file.name +
      "  ·  " + (file.size / 1048576).toFixed(1) + " MB";
    $("filenameStrip").hidden = false;
    analyzeBtn.disabled = false;
  }

  fileInput.addEventListener("change", function () { setFile(this.files[0]); });

  ["dragenter", "dragover"].forEach(function (ev) {
    dropzone.addEventListener(ev, function (e) {
      e.preventDefault(); dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    dropzone.addEventListener(ev, function (e) {
      e.preventDefault(); dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", function (e) {
    if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });

  /* ══════════════════ ANALYZE ══════════════════ */
  function showError(msg) { var b = $("errorBox"); b.textContent = msg; b.hidden = false; }
  function hideError() { $("errorBox").hidden = true; }

  analyzeBtn.addEventListener("click", function () {
    if (!selectedFile) return;
    hideError();
    $("emptyState").hidden = true;
    $("results").hidden = true;
    $("loadingBox").hidden = false;
    analyzeBtn.disabled = true;

    var fd = new FormData();
    fd.append("image", selectedFile);

    apiFetch("/api/analyze", { method: "POST", body: fd })
      .then(function (res) {
        return res.json().then(function (data) {
          if (!res.ok) throw new Error(data.error || ("Request failed (" + res.status + ")"));
          return data;
        });
      })
      .then(function (data) {
        render(data);
        if (isGuest()) loadViewer();
      })
      .catch(function (err) {
        showError(err.message || "Analysis failed.");
        $("emptyState").hidden = false;
      })
      .finally(function () {
        $("loadingBox").hidden = true;
        analyzeBtn.disabled = false;
      });
  });

  /* ══════════════════ RENDERING ══════════════════ */
  function titleCase(s) {
    return String(s).replace(/_/g, " ").replace(/\b\w/g, function (c) {
      return c.toUpperCase();
    });
  }

  function confBadge(conf) {
    var pct = Math.round(conf * 100);
    return conf >= 0.70
      ? { cls: "conf-high", text: "✓ " + pct + "% High", color: "#16A34A" }
      : { cls: "conf-med",  text: "◑ " + pct + "% Medium", color: "#D97706" };
  }

  /* Cards are patched in place rather than re-rendered: a re-render would tear
     the slider out from under the pointer mid-drag. */
  function macroRow(item) {
    if (item.has_data === false) {
      return '<div class="food-macro-item">No nutrition data available</div>';
    }
    return '<div class="food-macro-item"><span>' + item.calories.toFixed(0) + '</span> kcal</div>' +
           '<div class="food-macro-item"><span>' + item.protein.toFixed(1) + 'g</span> protein</div>' +
           '<div class="food-macro-item"><span>' + item.carbs.toFixed(1) + 'g</span> carbs</div>' +
           '<div class="food-macro-item"><span>' + item.fat.toFixed(1) + 'g</span> fat</div>';
  }

  /* Secondary line: portion total, the serving size behind it, and the density. */
  function densityLine(item) {
    if (item.has_data === false || item.calories_per_100g == null) return "";
    var serving = "";
    if (item.serving_g != null) {
      var grams = Math.round(item.serving_g) + "g";
      var qty = item.quantity || 1;
      serving = item.countable
        ? " (" + qty + " &times; " + grams + ")"
        : " (" + grams + " serving)";
    }
    return item.calories.toFixed(0) + " kcal total" + serving +
           " &nbsp;&middot;&nbsp; " + item.calories_per_100g.toFixed(0) + " kcal/100g";
  }

  /* How many pieces were eaten is the user's call: the detector's region count
     both over- and under-counts, so it never multiplies anything. */
  function quantityControl(item, idx) {
    if (!item.countable || item.has_data === false || !item.serving_g) return "";
    return '' +
      '<div class="qty-row" data-idx="' + idx + '">' +
      '  <span class="portion-label">Quantity</span>' +
      '  <input type="number" class="portion-input qty-input" id="pq' + idx + '"' +
      '         min="1" max="' + MAX_QTY + '" step="1" value="1"' +
      '         aria-label="Number of pieces">' +
      '</div>';
  }

  /* The serving weight is a fixed per-class estimate — the pipeline cannot measure
     portions from a photo — so it is exposed and adjustable, a quarter to triple. */
  function portionControl(item, idx) {
    if (item.has_data === false || !item.serving_g) return "";
    var std = Math.round(item.serving_g);
    var min = Math.max(5, Math.round(std * 0.25));
    var max = Math.round(std * 3);
    return quantityControl(item, idx) +
      '<div class="portion-row" data-idx="' + idx + '">' +
      '  <span class="portion-label">' + (item.countable ? "Per item" : "Portion") + '</span>' +
      '  <input type="range" class="portion-slider" id="pr' + idx + '"' +
      '         min="' + min + '" max="' + max + '" step="1" value="' + std + '"' +
      '         aria-label="Portion size in grams">' +
      '  <span class="portion-value">' +
      '    <input type="number" class="portion-input" id="pn' + idx + '"' +
      '           min="1" max="3000" step="5" value="' + std + '"' +
      '           aria-label="Portion size in grams">g' +
      '  </span>' +
      '  <button type="button" class="portion-reset" id="pz' + idx + '" disabled' +
      '          title="Back to the ' + std + 'g standard serving">Reset</button>' +
      '</div>' +
      '<p class="portion-note">Standard serving estimate &mdash; ' +
      'adjust if your portion differs.</p>';
  }

  function foodCard(item, idx) {
    var name = titleCase(item.name);
    var regions = item.regions || 1;
    var badge = confBadge(item.confidence);
    var src = item.source === "api" ? "USDA API" : "local fallback";
    var matched = (item.matched || "—").slice(0, 55);
    var density = densityLine(item);

    return '' +
      '<div class="food-card" style="border-left-color:' + badge.color + '">' +
      '  <div class="food-card-header">' +
      '    <p class="food-card-name">' + escapeHtml(name) + '</p>' +
      '    <span class="conf-badge ' + badge.cls + '">' + badge.text + '</span>' +
      '  </div>' +
      '  <div class="food-macros" id="macros' + idx + '">' + macroRow(item) + '</div>' +
      '  <p class="food-density" id="density' + idx + '"' + (density ? "" : " hidden") +
      '>' + density + '</p>' +
      portionControl(item, idx) +
      '  <p class="food-source">Detected in ' + regions + ' region' + (regions > 1 ? "s" : "") +
      ' &nbsp;&middot;&nbsp; ' + src + ' &nbsp;&middot;&nbsp; ' + escapeHtml(matched) + '</p>' +
      '</div>';
  }

  function unrecognizedCard(u) {
    var pct = Math.round(u.confidence * 100);
    return '' +
      '<div class="food-card food-card-unrecog">' +
      '  <div class="food-card-header">' +
      '    <p class="food-card-name">Unrecognized food</p>' +
      '    <span class="conf-badge conf-low">&#9888; ' + pct + '% Below threshold</span>' +
      '  </div>' +
      '  <p class="unrecog-note">Outside the 20 trained classes &nbsp;&middot;&nbsp; ' +
      '    top guess: &ldquo;' + escapeHtml(String(u.top_guess).replace(/_/g, " ")) + '&rdquo;</p>' +
      '</div>';
  }

  function metricCards(t) {
    var defs = [
      ["Total calories", t.calories.toFixed(0), " kcal", "#2563EB"],
      ["Protein",        t.protein.toFixed(1),  "g",     "#16A34A"],
      ["Carbohydrates",  t.carbs.toFixed(1),    "g",     "#D97706"],
      ["Fat",            t.fat.toFixed(1),      "g",     "#DC2626"]
    ];
    return defs.map(function (d) {
      return '<div class="metric-card" style="border-left-color:' + d[3] + '">' +
             '  <p class="metric-card-label">' + d[0] + '</p>' +
             '  <p class="metric-card-value">' + d[1] +
             '<span class="metric-card-unit">' + d[2] + '</span></p>' +
             '</div>';
    }).join("");
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function render(data) {
    var rec = data.recognized || [];
    var unrec = data.unrecognized || [];
    var totals = data.totals || { calories: 0, protein: 0, carbs: 0, fat: 0 };

    if (!rec.length && !unrec.length) {
      showError(data.message || "No food regions detected. Try a clearer photo or a closer crop.");
      $("emptyState").hidden = false;
      return;
    }

    $("annotatedImg").src = "data:image/png;base64," + data.annotated_image;
    $("detectCaption").textContent =
      data.region_count + " region(s) detected · " +
      rec.length + " recognised · " + unrec.length + " unrecognised";

    // Kept as returned; portions[] holds the user's corrections on top of it.
    analysis = data;
    portions = rec.map(function (item) {
      return item.has_data === false ? null : (item.serving_g || null);
    });
    quantities = rec.map(function () { return 1; });

    $("metricGrid").innerHTML = rec.length ? metricCards(totals) : "";
    $("metricGrid").hidden = !rec.length;

    $("cardColumn").innerHTML =
      rec.map(foodCard).join("") + unrec.map(unrecognizedCard).join("");

    $("results").hidden = false;

    if (rec.length) {
      startChat(data);
      applyPortions(true);
    } else {
      $("chartSection").hidden = true;
      $("goalSection").hidden = true;
      $("warnSection").hidden = true;
      $("chatSection").hidden = true;
    }
  }

  /* Charts are server-rendered PNGs, so this page, the Streamlit build and the
     thesis figures share one implementation. Dragging a slider would fire a
     request per pixel, so redraws are debounced and stale replies dropped. */
  var chartSeq = 0;
  var chartTimer = null;

  function scheduleCharts(items, totals, immediate) {
    clearTimeout(chartTimer);
    if (immediate) {
      drawCharts(items, totals);
    } else {
      chartTimer = setTimeout(function () { drawCharts(items, totals); }, 250);
    }
  }

  function drawCharts(items, totals) {
    var seq = ++chartSeq;

    $("pieChart").src = "/api/chart/macro-pie?protein=" + totals.protein.toFixed(2) +
                        "&carbs=" + totals.carbs.toFixed(2) +
                        "&fat=" + totals.fat.toFixed(2);

    apiFetch("/api/chart/calorie-bar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ recognized: items })
    })
      .then(function (res) { return res.ok ? res.blob() : null; })
      .then(function (blob) {
        if (!blob || seq !== chartSeq) return;   // a later portion change won
        var bar = $("barChart");
        if (bar.dataset.url) URL.revokeObjectURL(bar.dataset.url);
        var url = URL.createObjectURL(blob);
        bar.dataset.url = url;
        bar.src = url;
      })
      .catch(function () { /* the previous chart stays on screen */ });

    $("chartSection").hidden = false;
  }

  function renderGoal(kcal) {
    var pct = Math.min(kcal / DAILY_GOAL, 1) * 100;
    var pctDisp = Math.round(pct);
    var remaining = Math.max(DAILY_GOAL - kcal, 0);
    var over = kcal > DAILY_GOAL;

    $("goalCard").className = "goal-card" + (over ? " goal-over" : "");
    $("goalCard").innerHTML =
      '<div class="goal-header">' +
      '  <p class="goal-label">Progress toward ' + DAILY_GOAL + ' kcal daily goal</p>' +
      '  <span class="goal-pct">' + pctDisp + '%</span>' +
      '</div>' +
      '<div class="goal-track"><div class="goal-fill" style="width:' + pctDisp + '%"></div></div>' +
      '<div class="goal-sub">' +
      '  <span><strong>' + kcal.toFixed(0) + ' kcal</strong> consumed</span>' +
      '  <span><strong>' + remaining.toFixed(0) + ' kcal</strong> remaining</span>' +
      '</div>';
    $("goalSection").hidden = false;
  }


  /* ══════════════════ NUTRITIONAL FLAGS ══════════════════
     Descriptive, not diagnostic: each row states the measured number and the
     reference value it is compared against, computed from the totals
     /api/analyze already returned. */
  var WARN_CARB_G     = 60;    /* ~a third of a typical 200-250g day */
  var WARN_FAT_G      = 35;    /* ~half of a typical 65-70g day */
  var WARN_KCAL_SHARE = 0.4;   /* of the daily calorie goal */

  /* Lucide "triangle-alert" and "circle-check". */
  var ICON_WARN =
    '<svg class="warn-icon" viewBox="0 0 24 24" width="18" height="18"' +
    ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"' +
    ' stroke-linejoin="round" aria-hidden="true">' +
    '<path d="m21.7 18-8-14a2 2 0 0 0-3.4 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3Z"></path>' +
    '<path d="M12 9v4"></path><path d="M12 17h.01"></path></svg>';
  var ICON_OK =
    '<svg class="warn-icon" viewBox="0 0 24 24" width="18" height="18"' +
    ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"' +
    ' stroke-linejoin="round" aria-hidden="true">' +
    '<circle cx="12" cy="12" r="10"></circle><path d="m9 12 2 2 4-4"></path></svg>';

  function warnRow(title, sub, ok) {
    return '<div class="warn-row' + (ok ? " warn-row-ok" : "") + '">' +
           (ok ? ICON_OK : ICON_WARN) +
           '<div>' +
           '<p class="warn-row-title">' + title + '</p>' +
           '<p class="warn-row-sub">' + sub + '</p>' +
           '</div></div>';
  }

  function renderWarnings(totals) {
    var kcalCap = Math.round(DAILY_GOAL * WARN_KCAL_SHARE);
    var rows = [];

    if (totals.carbs > WARN_CARB_G) {
      rows.push(warnRow(
        "High carbohydrate content \u2014 " + Math.round(totals.carbs) + "g in this meal",
        "Reference point: " + WARN_CARB_G + "g, about a third of a typical " +
        "200\u2013250g daily carbohydrate intake."));
    }
    if (totals.fat > WARN_FAT_G) {
      rows.push(warnRow(
        "High fat content \u2014 " + Math.round(totals.fat) + "g in this meal",
        "Reference point: " + WARN_FAT_G + "g, about half of a typical " +
        "65\u201370g daily fat intake."));
    }
    if (totals.calories > kcalCap) {
      rows.push(warnRow(
        "High calorie content \u2014 " + Math.round(totals.calories) + " kcal in this meal",
        "Reference point: " + kcalCap + " kcal, " + Math.round(WARN_KCAL_SHARE * 100) +
        "% of the " + DAILY_GOAL + " kcal daily goal used on this page."));
    }

    if (!rows.length) {
      rows.push(warnRow(
        "Balanced against the reference values",
        "Carbohydrate, fat and calorie totals for this meal all stay within the " +
        "reference points used here (" + WARN_CARB_G + "g carbs, " + WARN_FAT_G +
        "g fat, " + kcalCap + " kcal).", true));
    }

    $("warnCard").innerHTML = rows.join("") +
      '<p class="warn-note">Informational only. These flags compare the ' +
      'analysed meal against general reference values for a single meal and ' +
      'are not medical or dietary advice; portion sizes are estimates. ' +
      'Speak to a doctor or dietitian about your own needs.</p>';
    $("warnSection").hidden = false;
  }

  /* ══════════════════ PORTION ADJUSTMENT ══════════════════
     The fixed per-class serving weight is the largest source of error in the
     calorie figure, so it is adjustable. Everything scales linearly from per-100g
     values the browser already has, so a correction costs no round trip. */
  var analysis = null;    // the last /api/analyze response, unmodified
  var portions = [];      // chosen grams per recognised item, index-aligned
  var quantities = [];    // pieces per recognised item; stays 1 for uncountables
  var MAX_QTY = 50;

  /* Macro densities follow from the serving weight by the same linear relation. */
  function per100(item) {
    if (item.has_data === false || !item.serving_g) return null;
    var k = 100 / item.serving_g;
    return {
      calories: item.calories_per_100g != null
        ? item.calories_per_100g
        : (item.calories || 0) * k,
      protein: (item.protein || 0) * k,
      carbs:   (item.carbs   || 0) * k,
      fat:     (item.fat     || 0) * k
    };
  }

  function adjustedItem(item, idx) {
    var per = per100(item);
    var grams = portions[idx];
    if (!per || grams == null) return item;      // nothing to scale it by
    var qty = item.countable ? (quantities[idx] || 1) : 1;
    var factor = grams * qty / 100;
    return {
      name:       item.name,
      regions:    item.regions,
      countable:  item.countable,
      quantity:   qty,
      confidence: item.confidence,
      source:     item.source,
      matched:    item.matched,
      has_data:   item.has_data,
      serving_g:  grams,
      standard_g: item.serving_g,
      adjusted:   grams !== item.serving_g,
      calories:   per.calories * factor,
      protein:    per.protein  * factor,
      carbs:      per.carbs    * factor,
      fat:        per.fat      * factor,
      calories_per_100g: per.calories
    };
  }

  function sumTotals(items) {
    var t = { calories: 0, protein: 0, carbs: 0, fat: 0 };
    items.forEach(function (i) {
      t.calories += i.calories || 0;
      t.protein  += i.protein  || 0;
      t.carbs    += i.carbs    || 0;
      t.fat      += i.fat      || 0;
    });
    return t;
  }

  /* One place that pushes the current portions through every dependent view. */
  function applyPortions(immediate) {
    if (!analysis) return;
    var items = (analysis.recognized || []).map(adjustedItem);
    var totals = sumTotals(items);

    items.forEach(function (item, idx) {
      var macros = $("macros" + idx);
      if (macros) macros.innerHTML = macroRow(item);
      var density = $("density" + idx);
      if (density) {
        var line = densityLine(item);
        density.innerHTML = line;
        density.hidden = !line;
      }
    });

    $("metricGrid").innerHTML = metricCards(totals);
    renderGoal(totals.calories);
    renderWarnings(totals);
    updateChatContext(items, totals);
    scheduleCharts(items, totals, immediate);
  }

  function setPortion(idx, grams, source) {
    if (!analysis || !analysis.recognized[idx]) return;
    portions[idx] = grams;

    var slider = $("pr" + idx), field = $("pn" + idx), reset = $("pz" + idx);
    if (slider && slider !== source) slider.value = grams;   // clamps to its range
    if (field && field !== source) field.value = grams;

    var std = Math.round(analysis.recognized[idx].serving_g);
    if (reset) reset.disabled = grams === std;
    var row = document.querySelector('.portion-row[data-idx="' + idx + '"]');
    if (row) row.classList.toggle("is-adjusted", grams !== std);

    applyPortions(false);
  }

  function onPortionInput(e) {
    var el = e.target;
    if (!el || !el.classList) return;
    if (!el.classList.contains("portion-slider") &&
        !el.classList.contains("portion-input")) return;

    var row = el.parentNode;
    while (row && !(row.classList && row.classList.contains("portion-row"))) {
      row = row.parentNode;
    }
    if (!row) return;
    var idx = parseInt(row.dataset.idx, 10);

    var val = parseFloat(el.value);
    if (!isFinite(val) || val <= 0) {
      // Mid-typing an empty box is not an error; only a committed one is.
      if (e.type === "change") el.value = portions[idx];
      return;
    }
    setPortion(idx, Math.min(3000, Math.max(1, Math.round(val))), el);
  }

  function onPortionClick(e) {
    var el = e.target;
    if (!el || !el.classList || !el.classList.contains("portion-reset")) return;
    var idx = parseInt(el.id.replace("pz", ""), 10);
    if (!analysis || !analysis.recognized[idx]) return;
    setPortion(idx, Math.round(analysis.recognized[idx].serving_g), null);
  }

  function onQuantityInput(e) {
    var el = e.target;
    if (!el || !el.classList || !el.classList.contains("qty-input")) return;
    var idx = parseInt(el.id.replace("pq", ""), 10);
    if (!analysis || !analysis.recognized[idx]) return;

    var val = parseInt(el.value, 10);
    if (!isFinite(val) || val < 1) {
      if (e.type === "change") el.value = quantities[idx];
      return;
    }
    val = Math.min(MAX_QTY, val);
    if (e.type === "change") el.value = val;   // drops a typed fraction or overshoot
    quantities[idx] = val;
    applyPortions(false);
  }

  /* Bound once on the container that outlives every re-render, so listeners never stack. */
  $("cardColumn").addEventListener("input", onPortionInput);
  $("cardColumn").addEventListener("change", onPortionInput);
  $("cardColumn").addEventListener("input", onQuantityInput);
  $("cardColumn").addEventListener("change", onQuantityInput);
  $("cardColumn").addEventListener("click", onPortionClick);

  /* ══════════════════ CHAT (Gemini) ══════════════════ */
  /* The meal context travels with every message — the server keeps no session
     state, so a reload or a second tab can never answer about another plate. */
  var chatContext = null;
  var chatHistory = [];
  var chatBusy = false;
  var chatKeyChecked = false;

  var chatLog = $("chatLog");
  var chatInput = $("chatInput");
  var chatSend = $("chatSend");

  function startChat(data) {
    chatContext = {
      recognized:   data.recognized || [],
      totals:       data.totals || {},
      unrecognized: data.unrecognized || []
    };
    chatHistory = [];
    chatLog.innerHTML = "";
    hideChatError();
    $("chatTyping").hidden = true;
    chatInput.value = "";
    setChatEnabled(true);
    $("chatSection").hidden = false;
    renderChatAccess();
    if (!isGuest()) checkChatKey();
  }

  /* The assistant is login-only; guests see a prompt to log in instead. */
  function renderChatAccess() {
    var locked = isGuest();
    $("chatLocked").hidden = !locked;
    $("chatChips").hidden = locked;
    $("chatOpen").hidden = locked;
  }

  /* Portion corrections must reach Gemini too, without wiping the conversation. */
  function updateChatContext(items, totals) {
    if (!chatContext) return;
    chatContext.recognized = items;
    chatContext.totals = totals;
  }

  function setChatEnabled(on) {
    chatBusy = !on;
    chatInput.disabled = !on;
    chatSend.disabled = !on;
    document.querySelectorAll(".chat-chip").forEach(function (c) {
      c.disabled = !on;
    });
  }

  function showChatError(msg) {
    var b = $("chatErrorBox"); b.textContent = msg; b.hidden = false;
  }
  function hideChatError() { $("chatErrorBox").hidden = true; }

  /* One status check per session: say up front when no key is configured. */
  function checkChatKey() {
    if (chatKeyChecked) return;
    chatKeyChecked = true;
    apiFetch("/api/chat/status")
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (st) {
        if (st && st.available === false) {
          showChatError("Chat is unavailable: no Gemini API key found. Set " +
                        (st.key_name || "GEMINI_API_KEY") + " in src/.env" +
                        " and reload this page — no restart needed.");
          setChatEnabled(false);
        }
      })
      .catch(function () { /* the first question will surface any problem */ });
  }

  /* Gemini answers come back as light Markdown. Everything is HTML-escaped
     first, so only the tags this function emits can ever reach innerHTML. */
  function mdInline(t) {
    return t
      .replace(/\*\*([^*]+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+?)\*(?!\*)/g, "$1<em>$2</em>");
  }

  function renderMarkdown(raw) {
    var lines = escapeHtml(String(raw == null ? "" : raw))
                  .replace(/\r\n/g, "\n").split("\n");
    var html = "", para = [], list = null;

    function flushPara() {
      if (!para.length) return;
      html += "<p>" + mdInline(para.join("<br>")) + "</p>";
      para = [];
    }
    function flushList() {
      if (!list) return;
      html += "<ul>" + list.join("") + "</ul>";
      list = null;
    }

    lines.forEach(function (line) {
      var bullet = /^\s*[-*+]\s+(.*)$/.exec(line);
      if (bullet) {                       // "- item" / "* item"
        flushPara();
        if (!list) list = [];
        list.push("<li>" + mdInline(bullet[1].trim()) + "</li>");
      } else if (!line.trim()) {          // blank line ends the block
        flushPara();
        flushList();
      } else {
        flushList();
        para.push(line.trim());
      }
    });
    flushPara();
    flushList();
    return html;
  }

  function appendBubble(role, text) {
    var div = document.createElement("div");
    div.className = "chat-msg " + (role === "user" ? "chat-msg-user" : "chat-msg-bot");
    if (role === "user") {
      div.textContent = text;             // the user's own words, verbatim
    } else {
      div.innerHTML = renderMarkdown(text);
    }
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
    return div;
  }

  function sendChat(text) {
    text = (text || "").trim();
    if (!text || chatBusy || !chatContext) return;

    hideChatError();
    appendBubble("user", text);
    chatInput.value = "";
    setChatEnabled(false);
    $("chatTyping").hidden = false;

    apiFetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        context: chatContext,
        history: chatHistory
      })
    })
      .then(function (res) {
        return res.json().then(function (data) {
          if (!res.ok) throw new Error(data.error || ("Chat failed (" + res.status + ")"));
          return data;
        });
      })
      .then(function (data) {
        appendBubble("assistant", data.reply);
        chatHistory.push({ role: "user", text: text });
        chatHistory.push({ role: "assistant", text: data.reply });
      })
      .catch(function (err) {
        showChatError(err.message || "Could not reach the assistant.");
      })
      .finally(function () {
        $("chatTyping").hidden = true;
        setChatEnabled(true);
        chatInput.focus();
      });
  }

  $("chatForm").addEventListener("submit", function (e) {
    e.preventDefault();
    sendChat(chatInput.value);
  });

  document.querySelectorAll(".chat-chip").forEach(function (chip) {
    chip.addEventListener("click", function () { sendChat(chip.textContent); });
  });


  /* ══════════════════ DATASET ANALYSIS ══════════════════
     Nutrition across all 20 classes, no uploaded photo involved. The server caches
     the USDA lookups; Refresh re-fetches them (?refresh=1). */
  var DATASET_CHARTS = [
    ["dsScatter",  "/api/chart/dataset/calorie-protein"],
    ["dsMacro",    "/api/chart/dataset/macro-composition"],
    ["dsDensity",  "/api/chart/dataset/calorie-density"]
  ];
  var datasetLoaded = false;

  function loadDataset(refresh) {
    if (datasetLoaded && !refresh) return;
    datasetLoaded = true;

    var btn = $("datasetRefresh");
    btn.disabled = true;
    $("datasetRefreshText").textContent = refresh ? "Refreshing…" : "Refresh";
    hideDatasetError();

    /* On a refresh the JSON call goes first and the charts wait for it: refreshing
       all four endpoints at once would be four times 20 live USDA lookups. A normal
       load is a cache read, so those can go together. */
    function drawCharts(bust) {
      DATASET_CHARTS.forEach(function (pair) {
        $(pair[0]).src = pair[1] + (bust ? "?t=" + bust : "");
      });
    }
    if (!refresh) drawCharts(null);

    apiFetch("/api/dataset-nutrition" + (refresh ? "?refresh=1" : ""))
      .then(function (res) {
        return res.json().then(function (data) {
          if (!res.ok) throw new Error(data.error || "Could not load nutrition data.");
          return data;
        });
      })
      .then(function (data) {
        if (refresh) drawCharts(Date.now());
        datasetCaptions(data);
      })
      .catch(function (err) {
        datasetLoaded = false;              // let the next visit try again
        var box = $("datasetErrorBox");
        box.textContent = err.message;
        box.hidden = false;
      })
      .finally(function () {
        btn.disabled = false;
        $("datasetRefreshText").textContent = "Refresh";
      });
  }

  function hideDatasetError() { $("datasetErrorBox").hidden = true; }

  function datasetCaptions(data) {
    var rows = (data.classes || []).filter(function (r) {
      return r.calories_per_100g;
    });
    if (!rows.length) return;

    var byDensity = rows.slice().sort(function (a, b) {
      return b.calories_per_100g - a.calories_per_100g;
    });
    var byProtein = rows.slice().sort(function (a, b) {
      return (b.protein_per_100g || 0) - (a.protein_per_100g || 0);
    });
    var withMacros = rows.filter(function (r) { return r.macro_pct; });
    var byCarb = withMacros.slice().sort(function (a, b) {
      return b.macro_pct.carbs - a.macro_pct.carbs;
    });
    var mean = byDensity.reduce(function (s, r) {
      return s + r.calories_per_100g;
    }, 0) / byDensity.length;

    function name(r) { return escapeHtml(r.label); }
    function kcal(r) { return Math.round(r.calories_per_100g); }

    $("dsScatterCaption").innerHTML =
      "Calories and protein per 100g, one point per class. " +
      "The protein-rich corner is meat-led — " + name(byProtein[0]) + " at " +
      byProtein[0].protein_per_100g + "g protein per 100g — while the " +
      "desserts and fried sides sit low and to the right: plenty of calories, " +
      "little protein. Colour shows which macro supplies most of the energy.";

    $("dsMacroCaption").innerHTML =
      "Each bar is one class split into protein, carbohydrate and fat as a " +
      "share of its calories, sorted by carbohydrate share. Percentages rather " +
      "than grams, so a 60g samosa and a 250g curry stay comparable. " +
      "Most carb-led: " + name(byCarb[0]) + " (" +
      Math.round(byCarb[0].macro_pct.carbs) + "% of calories); least: " +
      name(byCarb[byCarb.length - 1]) + " (" +
      Math.round(byCarb[byCarb.length - 1].macro_pct.carbs) + "%).";

    $("dsDensityCaption").innerHTML =
      "Calories per 100g, densest first. " + name(byDensity[0]) + " leads at " +
      kcal(byDensity[0]) + " kcal/100g and " +
      name(byDensity[byDensity.length - 1]) + " trails at " +
      kcal(byDensity[byDensity.length - 1]) + " — a " +
      (byDensity[0].calories_per_100g /
       byDensity[byDensity.length - 1].calories_per_100g).toFixed(1) +
      "&times; spread. Amber bars are above the " + Math.round(mean) +
      " kcal average (dashed line).";

    $("datasetMeta").innerHTML =
      "Nutrition for all " + data.n_classes + " food classes the model knows, " +
      "per 100g. " + data.from_api + " looked up from the USDA database" +
      (data.from_fallback
        ? " and " + data.from_fallback + " from the built-in table"
        : "") +
      ", then cached &mdash; nothing here depends on an uploaded photo.";
  }

  $("datasetRefresh").addEventListener("click", function () {
    loadDataset(true);
  });

  /* ══════════════════ TRAINING VIEW ══════════════════ */
  var trainingRuns = null;
  var trainingLoading = false;

  function loadTrainingRuns() {
    if (trainingRuns || trainingLoading) return;
    trainingLoading = true;

    apiFetch("/api/training-runs")
      .then(function (res) {
        return res.json().then(function (data) {
          if (!res.ok) throw new Error(data.error || "Could not load training history.");
          return data;
        });
      })
      .then(function (data) {
        trainingRuns = data.runs || [];
        var sel = $("runSelect");
        sel.innerHTML = trainingRuns.map(function (r) {
          return '<option value="' + r.id + '">' + escapeHtml(r.label) + "</option>";
        }).join("");
        sel.value = trainingRuns.length ? trainingRuns[trainingRuns.length - 1].id : "";
        sel.addEventListener("change", function () { showRun(this.value); });
        showRun(sel.value);
      })
      .catch(function (err) {
        var b = $("trainErrorBox");
        b.textContent = err.message;
        b.hidden = false;
      })
      .finally(function () { trainingLoading = false; });
  }

  /* Loaded lazily with the training view so the analysis page never waits on it;
     the server caches the evaluation, so this is a plain file read. */
  var perClassLoaded = false;

  function loadPerClassAccuracy() {
    if (perClassLoaded) return;
    perClassLoaded = true;
    $("perClassChart").src = "/api/chart/per-class-accuracy";

    apiFetch("/api/per-class-accuracy")
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || !data.classes || !data.classes.length) return;
        var worst = data.classes.slice(0, 3).map(function (c) {
          var target = c.confused_with
            ? " (→ " + titleCase(c.confused_with).toLowerCase() + ")"
            : "";
          return titleCase(c.name).toLowerCase() + " " +
                 (c.accuracy * 100).toFixed(1) + "%" + target;
        }).join(", ");

        $("perClassCaption").innerHTML =
          "Weakest classes: " + escapeHtml(worst) + ". Dashed line = overall " +
          (data.overall_accuracy * 100).toFixed(1) + "% across " +
          data.n_test_images.toLocaleString() + " held-out test images. " +
          "Confusable pairs sit at the bottom — fried rice &harr; paella and " +
          "tacos &harr; falafel here, the same pattern steak/filet_mignon showed " +
          "in the 15-class run.";
      })
      .catch(function () { /* keep the static caption */ });
  }

  function showRun(id) {
    var run = (trainingRuns || []).filter(function (r) { return String(r.id) === String(id); })[0];
    if (!run) return;
    $("lossChart").src = "data:image/png;base64," + run.loss_chart;
    $("accChart").src = "data:image/png;base64," + run.acc_chart;
  }

  /* ══════════════════ ACCESS ══════════════════
     Everyone on this page is logged in or a guest — the server sends anyone
     else to /login. Tokens live in HttpOnly cookies; this code never sees them. */
  var viewer = { user: null, guest: null };

  /* fetch() for the API: sends the cookies, sends a plain-object body as JSON,
     and reacts to the access errors the server marks with a `code`. */
  function apiFetch(url, opts) {
    var o = {};
    for (var k in opts || {}) { o[k] = opts[k]; }
    o.credentials = "same-origin";
    o.headers = new Headers((opts && opts.headers) || {});
    if (o.body && Object.prototype.toString.call(o.body) === "[object Object]") {
      o.body = JSON.stringify(o.body);
      o.headers.set("Content-Type", "application/json");
    }
    return fetch(url, o).then(function (res) {
      if (res.status === 401) {
        res.clone().json().then(function (data) {
          if (data.code === "auth_required") window.location.replace("/login");
          else if (data.code === "guest_limit_reached") loadViewer();
        }).catch(function () { /* not JSON: leave it to the caller */ });
      }
      return res;
    });
  }
  window.FoodLens = { apiFetch: apiFetch };

  function isGuest() { return !viewer.user; }

  function loadViewer() {
    return fetch("/auth/session", { credentials: "same-origin" })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (!data.user && !data.guest) { window.location.replace("/login"); return; }
        viewer = { user: data.user, guest: data.guest };
        renderViewer();
      })
      .catch(function () { /* keep the header as is; the API still enforces access */ });
  }

  function renderViewer() {
    var user = viewer.user, guest = viewer.guest;
    $("authUser").hidden = !user;
    $("authGuest").hidden = !guest;
    $("authEmail").textContent = user ? user.email : "";
    $("authEmail").title = user ? user.email : "";
    if (guest) {
      $("guestText").textContent = "Guest · " + guest.remaining + " of " +
                                   guest.limit + " free analyses left";
      $("guestLogin").hidden = !window.AUTH_ENABLED;
      $("limitCount").textContent = guest.limit;
    }

    var limited = !!guest && guest.remaining <= 0;
    $("uploadCard").hidden = limited;
    $("limitCard").hidden = !limited;
    if (limited) $("emptyState").hidden = true;

    $("datasetRefresh").hidden = !user;     // guests always get the cached data
    renderChatAccess();
  }

  $("logoutBtn").addEventListener("click", function () {
    this.disabled = true;
    fetch("/auth/logout", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: "{}"
    })
      .catch(function () { /* cookies are cleared server-side regardless */ })
      .then(function () { window.location.replace("/login"); });
  });

  loadViewer();

  showView("analysis");
})();
