/* FirstLook PWA service worker: cache-first, everything local. Bump CACHE when assets change. */
const CACHE = "firstlook-v2";
const ASSETS = ["./", "index.html", "model.js", "models.json", "manifest.webmanifest", "icon.svg"];
self.addEventListener("install", (e) => e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting())));
self.addEventListener("activate", (e) => e.waitUntil(
  caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener("fetch", (e) => {
  if (new URL(e.request.url).origin !== self.location.origin) return; // never touch other origins
  e.respondWith(caches.match(e.request).then((r) => r || fetch(e.request)));
});
