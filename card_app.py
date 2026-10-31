"""Browse the scanned-card collection in a web browser.

    python card_app.py            # then open http://localhost:8000

Search by name, filter by card type and set, and sort by any column. Uses
only the standard library; the data lives in cards.db (see card_db.py).
"""
from __future__ import annotations

import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import card_db

PORT = 8000

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Card Collection</title>
<style>
  :root { --bg:#f6f7f9; --panel:#fff; --ink:#1d2330; --muted:#677084; --line:#e2e5eb;
          --accent:#3b6fe0; --accent-ink:#fff; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14171d; --panel:#1c2028; --ink:#e7eaf0; --muted:#98a1b3; --line:#2c323d;
            --accent:#6d97ff; --accent-ink:#0d1220; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width:1200px; margin:0 auto; padding:24px 16px 48px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 18px; }
  .filters { display:flex; flex-wrap:wrap; gap:10px; background:var(--panel);
             border:1px solid var(--line); border-radius:10px; padding:12px; }
  .filters label { display:flex; flex-direction:column; gap:4px; font-size:12px;
                   color:var(--muted); flex:1 1 160px; }
  input, select, button { font:inherit; color:var(--ink); background:var(--bg);
                          border:1px solid var(--line); border-radius:7px; padding:7px 9px; }
  button { cursor:pointer; }
  .filters button { align-self:flex-end; flex:0 0 auto; }
  .table-wrap { overflow-x:auto; margin-top:14px; background:var(--panel);
                border:1px solid var(--line); border-radius:10px; }
  table { border-collapse:collapse; width:100%; min-width:820px; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line);
           vertical-align:middle; }
  th { font-size:12px; color:var(--muted); font-weight:600; cursor:pointer;
       user-select:none; white-space:nowrap; }
  th.nosort { cursor:default; }
  th .arrow { color:var(--accent); }
  tr:last-child td { border-bottom:none; }
  td img { width:44px; border-radius:4px; display:block; }
  .thumb { width:44px; height:61px; border-radius:4px; background:var(--line); }
  a { color:var(--accent); text-decoration:none; }
  a:hover { text-decoration:underline; }
  .pill { display:inline-block; padding:2px 8px; border-radius:99px; font-size:12px;
          background:var(--bg); border:1px solid var(--line); white-space:nowrap; }
  .qty { display:flex; align-items:center; gap:6px; }
  .qty button { padding:2px 8px; }
  .muted { color:var(--muted); }
  .empty { padding:40px; text-align:center; color:var(--muted); }
</style>
</head>
<body>
<main>
  <h1>Card Collection</h1>
  <p class="sub" id="summary">Loading...</p>
  <form class="filters" id="filters">
    <label>Name <input id="name" placeholder="e.g. Pikachu" autocomplete="off"></label>
    <label>Type <select id="type"><option value="">All types</option></select></label>
    <label>Set <select id="set"><option value="">All sets</option></select></label>
    <label>Stamp <select id="stamp"><option value="">Any</option>
      <option value="none">No stamp</option></select></label>
    <label>Sort by
      <select id="sort">
        <option value="name">Name</option><option value="type">Type</option>
        <option value="set">Set</option><option value="number">Set + number</option>
        <option value="rarity">Rarity</option><option value="stamp">Stamp</option>
        <option value="quantity">Quantity</option>
        <option value="scanned">Last scanned</option>
      </select></label>
    <label>Order
      <select id="desc"><option value="0">Ascending</option>
        <option value="1">Descending</option></select></label>
    <button type="button" id="reset">Reset</button>
  </form>
  <div class="table-wrap"><table>
    <thead><tr>
      <th class="nosort"></th><th data-sort="name">Name</th><th data-sort="type">Type</th>
      <th data-sort="set">Set</th><th data-sort="number">Number</th>
      <th data-sort="rarity">Rarity</th><th data-sort="stamp">Stamp</th>
      <th class="nosort">Energy</th>
      <th data-sort="quantity">Qty</th><th data-sort="scanned">Last scanned</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table></div>
</main>
<script>
const $ = id => document.getElementById(id);
// Limitless pictures come in _XS/_SM/_LG; the table only needs the small one.
const thumb = url => url.replace(/_SM\.png$/, "_XS.png");
const esc = s => String(s ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

async function loadChoices() {
  const c = await (await fetch("/api/choices")).json();
  // [select id, values, how many fixed options lead the list]
  for (const [id, list, fixed] of [["type", c.types, 1], ["set", c.sets, 1],
                                   ["stamp", c.stamps, 2]]) {
    const sel = $(id), keep = sel.value;
    sel.length = fixed;
    for (const v of list) sel.add(new Option(v, v));
    sel.value = list.includes(keep) || (fixed > 1 && keep === "none") ? keep : "";
  }
  $("summary").textContent = c.unique + " different cards, " + c.total + " in total";
}

async function loadCards() {
  const q = new URLSearchParams({name: $("name").value.trim(), type: $("type").value,
    set: $("set").value, stamp: $("stamp").value, sort: $("sort").value,
    desc: $("desc").value});
  const cards = await (await fetch("/api/cards?" + q)).json();
  document.querySelectorAll("th[data-sort]").forEach(th => {
    th.querySelector(".arrow")?.remove();
    if (th.dataset.sort === $("sort").value)
      th.insertAdjacentHTML("beforeend",
        ' <span class="arrow">' + ($("desc").value === "1" ? "&#9660;" : "&#9650;") + "</span>");
  });
  $("rows").innerHTML = cards.length ? cards.map(c => `
    <tr>
      <td>${c.image ? `<img src="${esc(thumb(c.image))}" alt="" loading="lazy">`
                    : '<div class="thumb"></div>'}</td>
      <td>${c.link ? `<a href="${esc(c.link)}" target="_blank" rel="noopener">${esc(c.name)}</a>`
                   : esc(c.name)}</td>
      <td><span class="pill">${esc(c.card_type)}</span></td>
      <td>${esc(c.set_name)} ${c.set_code ? `<span class="muted">(${esc(c.set_code)})</span>` : ""}</td>
      <td>${esc(c.number)}</td>
      <td>${esc(c.rarity)}</td>
      <td>${c.stamp ? `<span class="pill">${esc(c.stamp)}</span>` : ""}</td>
      <td>${esc(c.energy)}</td>
      <td><div class="qty">
        <button data-id="${c.id}" data-q="${c.quantity - 1}" title="Remove one">&minus;</button>
        <span>${c.quantity}</span>
        <button data-id="${c.id}" data-q="${c.quantity + 1}" title="Add one">+</button>
      </div></td>
      <td class="muted">${esc(c.last_scanned.replace("T", " "))}</td>
    </tr>`).join("")
    : '<tr><td colspan="10" class="empty">No cards match.</td></tr>';
}

function refresh() { loadChoices().then(loadCards); }

let typing;
$("name").addEventListener("input", () => { clearTimeout(typing); typing = setTimeout(loadCards, 200); });
for (const id of ["type", "set", "stamp", "sort", "desc"]) $(id).addEventListener("change", loadCards);
$("filters").addEventListener("submit", e => { e.preventDefault(); loadCards(); });
$("reset").addEventListener("click", () => {
  $("name").value = ""; $("type").value = ""; $("set").value = ""; $("stamp").value = "";
  $("sort").value = "name"; $("desc").value = "0"; loadCards();
});
document.querySelectorAll("th[data-sort]").forEach(th => th.addEventListener("click", () => {
  if ($("sort").value === th.dataset.sort) $("desc").value = $("desc").value === "1" ? "0" : "1";
  else { $("sort").value = th.dataset.sort; $("desc").value = "0"; }
  loadCards();
}));
$("rows").addEventListener("click", async e => {
  const b = e.target.closest("button[data-id]");
  if (!b) return;
  await fetch("/api/quantity", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({id: +b.dataset.id, quantity: +b.dataset.q})});
  refresh();
});
// New scans show up without reloading the page.
setInterval(refresh, 5000);
refresh();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(json.dumps(data).encode("utf-8"), "application/json", status)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif url.path == "/api/cards":
            self._json(card_db.search(q.get("name", ""), q.get("type", ""), q.get("set", ""),
                                      q.get("sort", "name"), q.get("desc") == "1",
                                      q.get("stamp", "")))
        elif url.path == "/api/choices":
            self._json(card_db.choices())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/quantity":
            self._json({"error": "not found"}, 404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            card_db.set_quantity(int(body["id"]), int(body["quantity"]))
        except (ValueError, KeyError, TypeError):
            self._json({"error": "bad request"}, 400)
            return
        self._json({"ok": True})

    def log_message(self, fmt, *args) -> None:
        pass                                # keep the terminal quiet


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = "http://localhost:{}".format(port)
    print("Card collection at {}  (Ctrl+C to stop)".format(url))
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
