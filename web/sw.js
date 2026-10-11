// Truth_Rail service worker: lets the app install on a phone and open its shell offline.
// Checks themselves always need the network, so /api, /whatsapp and /media are never cached.
const CACHE = "truth-rail-v1";
const SHELL = ["/", "/static/logo.svg", "/static/icon-192.png", "/manifest.webmanifest"];
self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  const skip = ["/api", "/whatsapp", "/media", "/health", "/qr.svg"];
  if (e.request.method !== "GET" || url.origin !== location.origin || skip.some(p => url.pathname.startsWith(p))) return;
  e.respondWith(
    fetch(e.request).then(r => {
      const copy = r.clone();
      caches.open(CACHE).then(c => c.put(e.request, copy));
      return r;
    }).catch(() => caches.match(e.request))
  );
});
