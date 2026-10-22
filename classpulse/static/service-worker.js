const CACHE_NAME = "classpulse-shell-v6";
const APP_SHELL = ["/login", "/static/dashboard.css", "/static/dashboard.js", "/static/theme.js"];

self.addEventListener("install", (event) => {
    event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)));
    self.skipWaiting();
});

self.addEventListener("activate", (event) => {
    event.waitUntil(
        caches.keys().then((keys) => Promise.all(keys.filter((key) => key.startsWith("classpulse-shell-") && key !== CACHE_NAME).map((key) => caches.delete(key))))
            .then(() => self.clients.claim())
    );
});

self.addEventListener("fetch", (event) => {
    const requestUrl = new URL(event.request.url);
    if (event.request.method !== "GET" || requestUrl.origin !== self.location.origin || requestUrl.pathname.startsWith("/api/") || requestUrl.pathname === "/overview") return;
    event.respondWith(fetch(event.request).then((response) => {
        if (response.ok && (requestUrl.pathname.startsWith("/static/") || requestUrl.pathname === "/login")) {
            const copy = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
        }
        return response;
    }).catch(() => caches.match(event.request)));
});
