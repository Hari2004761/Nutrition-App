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

  var TITLES = { analysis: "Culinary Intelligence", training: "Model Training Graphs" };

  function showView(name) {
    document.querySelectorAll(".sb-nav-item").forEach(function (li) {
      li.classList.toggle("active", li.dataset.view === name);
    });
    $("view-analysis").hidden = name !== "analysis";
    $("view-training").hidden = name !== "training";
    $("pageTitle").textContent = TITLES[name];
    if (name === "training") { loadTrainingRuns(); loadPerClassAccuracy(); }
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

    fetch("/api/analyze", { method: "POST", body: fd })
      .then(function (res) {
        return res.json().then(function (data) {
          if (!res.ok) throw new Error(data.error || ("Request failed (" + res.status + ")"));
          return data;
        });
      })
      .then(render)
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

  function foodCard(item) {
    var name = titleCase(item.name) + (item.count > 1 ? " ×" + item.count : "");
    var badge = confBadge(item.confidence);
    var src = item.source === "api" ? "USDA API" : "local fallback";
    var matched = (item.matched || "—").slice(0, 55);
    var macros = item.has_data === false
      ? '<div class="food-macro-item">No nutrition data available</div>'
      : '<div class="food-macro-item"><span>' + item.calories.toFixed(0) + '</span> kcal</div>' +
        '<div class="food-macro-item"><span>' + item.protein.toFixed(1) + 'g</span> protein</div>' +
        '<div class="food-macro-item"><span>' + item.carbs.toFixed(1) + 'g</span> carbs</div>' +
        '<div class="food-macro-item"><span>' + item.fat.toFixed(1) + 'g</span> fat</div>';

    // Secondary line: portion total, the serving size that produced it, and
    // the unscaled energy density that ties the two together.
    var density = "";
    if (item.has_data !== false && item.calories_per_100g != null) {
      var serving = "";
      if (item.serving_g != null) {
        serving = item.count > 1
          ? ' (' + item.count + ' &times; ' + item.serving_g + 'g serving)'
          : ' (' + item.serving_g + 'g serving)';
      }
      density = '<p class="food-density">' + item.calories.toFixed(0) +
                ' kcal total' + serving + ' &nbsp;&middot;&nbsp; ' +
                item.calories_per_100g.toFixed(0) + ' kcal/100g</p>';
    }

    return '' +
      '<div class="food-card" style="border-left-color:' + badge.color + '">' +
      '  <div class="food-card-header">' +
      '    <p class="food-card-name">' + escapeHtml(name) + '</p>' +
      '    <span class="conf-badge ' + badge.cls + '">' + badge.text + '</span>' +
      '  </div>' +
      '  <div class="food-macros">' + macros + '</div>' +
      density +
      '  <p class="food-source">' + src + ' &nbsp;&middot;&nbsp; ' + escapeHtml(matched) + '</p>' +
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

    $("metricGrid").innerHTML = rec.length ? metricCards(totals) : "";
    $("metricGrid").hidden = !rec.length;

    $("cardColumn").innerHTML =
      rec.map(foodCard).join("") + unrec.map(unrecognizedCard).join("");

    $("results").hidden = false;

    if (rec.length) {
      loadCharts(data);
      renderGoal(totals.calories);
      startChat(data);
    } else {
      $("chartSection").hidden = true;
      $("goalSection").hidden = true;
      $("chatSection").hidden = true;
    }
  }

  /* ── charts come from the server's matplotlib endpoints ── */
  function loadCharts(data) {
    var t = data.totals;
    var q = "?protein=" + t.protein + "&carbs=" + t.carbs + "&fat=" + t.fat;
    $("pieChart").src = "/api/chart/macro-pie" + q;

    fetch("/api/chart/calorie-bar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ recognized: data.recognized })
    })
      .then(function (res) { return res.ok ? res.blob() : null; })
      .then(function (blob) {
        if (!blob) return;
        var bar = $("barChart");
        if (bar.dataset.url) URL.revokeObjectURL(bar.dataset.url);
        var url = URL.createObjectURL(blob);
        bar.dataset.url = url;
        bar.src = url;
      });

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

  /* ══════════════════ CHAT (Gemini) ══════════════════ */
  /* The meal context travels with every message — the server keeps no
     session state, so a reload or a second tab can never answer about
     someone else's plate. */
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
    checkChatKey();
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

  /* One status check per session: if no key is configured, say so up front
     instead of letting the first question fail. */
  function checkChatKey() {
    if (chatKeyChecked) return;
    chatKeyChecked = true;
    fetch("/api/chat/status")
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (st) {
        if (st && st.available === false) {
          showChatError("Chat is unavailable: no Gemini API key found. Put " +
                        "your key in " + (st.key_file || "gemini_api_key.txt") +
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

    fetch("/api/chat", {
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

  /* ══════════════════ TRAINING VIEW ══════════════════ */
  var trainingRuns = null;
  var trainingLoading = false;

  function loadTrainingRuns() {
    if (trainingRuns || trainingLoading) return;
    trainingLoading = true;

    fetch("/api/training-runs")
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

  /* -- per-class accuracy: chart PNG + a caption naming the weakest classes --
     Loaded lazily with the training view so the analysis page never waits on
     it. The server caches the evaluation, so this is a plain file read. */
  var perClassLoaded = false;

  function loadPerClassAccuracy() {
    if (perClassLoaded) return;
    perClassLoaded = true;
    $("perClassChart").src = "/api/chart/per-class-accuracy";

    fetch("/api/per-class-accuracy")
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

  showView("analysis");
})();
