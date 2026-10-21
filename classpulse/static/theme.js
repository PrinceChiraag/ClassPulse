(function () {
    const storageKey = "classpulse-theme";
    const root = document.documentElement;
    const toggle = document.getElementById("theme-toggle");

    function applyTheme(theme, persist = false) {
        const isDark = theme === "dark";
        root.dataset.theme = isDark ? "dark" : "light";
        if (toggle) {
            toggle.setAttribute("aria-pressed", String(isDark));
            const icon = toggle.querySelector(".theme-icon");
            const label = toggle.querySelector(".theme-label");
            if (icon) icon.textContent = isDark ? "☀" : "☾";
            if (label) label.textContent = isDark ? "Day mode" : "Night mode";
        }
        if (persist) {
            try { localStorage.setItem(storageKey, isDark ? "dark" : "light"); } catch (_) { /* storage can be disabled */ }
        }
    }

    let savedTheme = "light";
    try { savedTheme = localStorage.getItem(storageKey) || "light"; } catch (_) { /* storage can be disabled */ }
    applyTheme(savedTheme);

    if (toggle) {
        toggle.addEventListener("click", () => {
            applyTheme(root.dataset.theme === "dark" ? "light" : "dark", true);
        });
    }
})();
