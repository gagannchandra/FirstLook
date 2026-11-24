/* FirstLook service worker: cache-first for the app shell, everything local. Bump CACHE when assets change.
   api/ requests (the optional local backend) always go to the network and are never cached. */
const CACHE = "firstlook-v19";
const ASSETS = ["./", "index.html", "css/app.css", "js/model.js", "js/charts.js", "js/app.js", "data/models.json", "data/dashboard.json",
  "manifest.webmanifest", "icon.svg",
  "fonts/atkinson-next-latin.woff2", "fonts/atkinson-next-latin-ext.woff2",
  "fonts/rethink-sans-latin.woff2", "fonts/rethink-sans-latin-ext.woff2"];
self.addEventListener("install", (e) => e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting())));
self.addEventListener("activate", (e) => e.waitUntil(
  caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.origin !== self.location.origin) return; // never touch other origins
  if (url.pathname.includes("/api/")) return; // live backend calls: network only
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then((r) => r || fetch(e.request)));
});
