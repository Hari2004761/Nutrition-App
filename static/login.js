/* FoodLens login page — log in, sign up, or continue as a guest.
   Tokens are set as HttpOnly cookies by Flask; this code never sees them. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var mode = "login";

  function postJSON(url, body) {
    return fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    }).then(function (res) {
      return res.json().then(function (data) {
        if (!res.ok) throw new Error(data.error || ("Request failed (" + res.status + ")"));
        return data;
      });
    });
  }

  function msg(kind, text) {
    $("authError").hidden = kind !== "error";
    $("authInfo").hidden = kind !== "info";
    if (kind) $(kind === "error" ? "authError" : "authInfo").textContent = text;
  }

  function setMode(m) {
    mode = m;
    var signup = m === "signup";
    document.querySelectorAll(".auth-tab").forEach(function (t) {
      var on = t.dataset.mode === m;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
    });
    $("authTitle").textContent = signup ? "Create an account" : "Log in to FoodLens";
    $("authSubmit").textContent = signup ? "Sign up" : "Log in";
    $("authPasswordInput").autocomplete = signup ? "new-password" : "current-password";
    msg(null);
  }

  document.querySelectorAll(".auth-tab").forEach(function (t) {
    t.addEventListener("click", function () { setMode(t.dataset.mode); });
  });

  $("authForm").addEventListener("submit", function (e) {
    e.preventDefault();
    var email = $("authEmailInput").value.trim();
    var password = $("authPasswordInput").value;
    if (!email || !password) { msg("error", "Enter your email and password."); return; }
    if (mode === "signup" && password.length < 6) {
      msg("error", "Password must be at least 6 characters."); return;
    }

    var submit = $("authSubmit");
    submit.disabled = true;
    msg(null);
    postJSON("/auth/" + mode, { email: email, password: password })
      .then(function (data) {
        if (data.confirmation_required) {
          setMode("login");
          msg("info", "Check your inbox for a confirmation link, then log in.");
          return;
        }
        window.location.replace("/");
      })
      .catch(function (err) {
        msg("error", err.message || "Could not reach the login server.");
      })
      .finally(function () { submit.disabled = false; });
  });

  var guestBtn = $("guestBtn");
  if (guestBtn) {
    guestBtn.addEventListener("click", function () {
      guestBtn.disabled = true;
      $("guestError").hidden = true;
      postJSON("/auth/guest")
        .then(function () { window.location.replace("/"); })
        .catch(function (err) {
          var box = $("guestError");
          box.textContent = err.message || "Could not start guest mode.";
          box.hidden = false;
          guestBtn.disabled = false;
        });
    });
  }
})();
