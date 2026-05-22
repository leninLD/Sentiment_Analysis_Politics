"""
wordcloud_component.py  (v2 — fixes blank canvas rendering)
────────────────────────────────────────────────────────────
Fixes applied vs v1:
  1. Removed broken SRI integrity hash that caused wordcloud2.js to silently fail
  2. Font loaded via FontFace API directly so we KNOW font is ready before drawing
  3. Retry loop — if wordcloud2 draws 0 words, retries up to 3x
  4. weightFactor uses sqrt for better visual spread
  5. System Devanagari fonts (Mangal, Arial Unicode MS) as offline fallbacks
  6. jsDelivr mirror as CDN fallback if cdnjs is blocked
  7. Graceful empty-state with styled message instead of blank box
"""

import re
import json
import unicodedata
from collections import Counter
import streamlit.components.v1 as components


# ── Stopwords ──────────────────────────────────────────────────────────────────
def _load_stopwords(path: str = "stop_words_nepali.txt") -> set:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {unicodedata.normalize("NFC", ln.strip())
                    for ln in f if ln.strip()}
    except FileNotFoundError:
        return set()

NEPALI_STOPWORDS = _load_stopwords("stop_words_nepali.txt")


# ── Suffix stripping ───────────────────────────────────────────────────────────
_SUFFIXES = re.compile(
    r'(हरुको|हरुका|हरुकी|हरुलाई|हरुले|हरुमा|हरुबाट|हरु'
    r'|लाई|बाट|सँग|सित|मार्फत|तिर|सम्म|भन्दा|भित्र|बाहिर'
    r'|को|का|की|ले|मा|मै|नै|चाहिँ|पनि|नि)$'
)

def _strip_suffix(word: str) -> str:
    m = _SUFFIXES.search(word)
    if m and m.start() >= 2:
        return word[:m.start()]
    return word


# ── Invisible / zero-width char remover ───────────────────────────────────────
_INVIS = re.compile(
    r'[\u00ad\u200b\u200c\u200d\u200e\u200f'
    r'\u202a\u202b\u202c\u202d\u202e\u202f'
    r'\u2060\u2061\u2062\u2063\u2064\ufeff]'
)

def _extract_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", str(text))
    text = _INVIS.sub("", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[@#]\S+', '', text)
    tokens = re.findall(r'[\u0900-\u097F]+', text)
    clean = []
    for tok in tokens:
        tok = unicodedata.normalize("NFC", tok).strip('\u0964\u0965')
        if not tok:
            continue
        root = _strip_suffix(tok)
        root = unicodedata.normalize("NFC", root)
        word = root if len(root) >= 2 else tok
        if len(word) < 2:
            continue
        if word in NEPALI_STOPWORDS:
            continue
        clean.append(word)
    return clean


def _build_frequencies(texts: list[str]) -> Counter:
    freq: Counter = Counter()
    for t in texts:
        if t and str(t).strip():
            freq.update(_extract_words(t))
    return freq


# ── Color palettes ─────────────────────────────────────────────────────────────
_PALETTES: dict[str, list[str]] = {
    "Blues":  ["#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8"],
    "Greens": ["#6ee7b7", "#34d399", "#10b981", "#059669", "#047857"],
    "Reds":   ["#fca5a5", "#f87171", "#ef4444", "#dc2626", "#b91c1c"],
}


# ── HTML template ──────────────────────────────────────────────────────────────
_TEMPLATE = """\
<!DOCTYPE html>
<html lang="ne">
<head>
<meta charset="utf-8">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
html, body {{ width:{width}px; height:{height}px; background:{bg}; overflow:hidden; }}
#wc {{ display:block; }}
#msg {{
  display:none; position:absolute; inset:0;
  align-items:center; justify-content:center;
  font-size:14px; color:#94a3b8;
  font-family:sans-serif; text-align:center; padding:1rem;
}}
</style>
</head>
<body>
<canvas id="wc" width="{width}" height="{height}"></canvas>
<div id="msg">Not enough Nepali text to build a word cloud.</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/wordcloud2.js/1.1.0/wordcloud2.min.js"></script>

<script>
var WORD_LIST = {word_list_json};
var COLORS    = {colors_json};
var BG        = "{bg}";
var W         = {width};
var H         = {height};

function showEmpty() {{
  document.getElementById("msg").style.display = "flex";
  var cv = document.getElementById("wc");
  var ctx = cv.getContext("2d");
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, W, H);
}}

function attemptDraw(retries) {{
  var canvas = document.getElementById("wc");
  var ctx    = canvas.getContext("2d");
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, W, H);

  if (!WORD_LIST || WORD_LIST.length === 0) {{ showEmpty(); return; }}

  var maxF = WORD_LIST[0][1];
  var minF = WORD_LIST[WORD_LIST.length - 1][1];
  var span = Math.max(maxF - minF, 1);

  WordCloud(canvas, {{
    list:            WORD_LIST,
    fontFamily:      "'Noto Sans Devanagari', 'Mangal', 'Arial Unicode MS', sans-serif",
    fontWeight:      "bold",
    color:           function() {{ return COLORS[Math.floor(Math.random() * COLORS.length)]; }},
    rotateRatio:     0,
    rotationSteps:   1,
    backgroundColor: BG,
    shrinkToFit:     true,
    drawOutOfBound:  false,
    shuffle:         true,
    minSize:         8,
    weightFactor:    function(size) {{
      return 14 + Math.sqrt((size - minF) / span) * 70;
    }},
    done: function() {{
      var d = ctx.getImageData(0, 0, W, H).data;
      var painted = false;
      for (var i = 0; i < d.length; i += 160) {{
        if (!(d[i] > 240 && d[i+1] > 240 && d[i+2] > 240)) {{ painted = true; break; }}
      }}
      if (!painted && retries < 4) {{
        setTimeout(function() {{ attemptDraw(retries + 1); }}, 500);
      }}
    }}
  }});
}}

function loadWC2ThenDraw() {{
  if (typeof WordCloud !== "undefined") {{
    attemptDraw(0);
  }} else {{
    var s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/wordcloud@1.1.0/src/wordcloud2.js";
    s.onload  = function() {{ attemptDraw(0); }};
    s.onerror = showEmpty;
    document.head.appendChild(s);
  }}
}}

if (window.FontFace) {{
  var font = new FontFace(
    "Noto Sans Devanagari",
    "url('https://fonts.gstatic.com/s/notosansdevanagari/v25/TuGoUUFzXI5FBtUq5a8bjKYTZjtB_FNvLJRGCZBOZA.woff2') format('woff2')",
    {{ weight: "700" }}
  );
  font.load()
    .then(function(f) {{ document.fonts.add(f); loadWC2ThenDraw(); }})
    .catch(function()  {{ loadWC2ThenDraw(); }});
}} else {{
  setTimeout(loadWC2ThenDraw, 900);
}}
</script>
</body>
</html>
"""


# ── Public API ─────────────────────────────────────────────────────────────────
def make_wordcloud_html(
    texts: list[str],
    colormap: str = "Blues",
    width: int = 660,
    height: int = 300,
    bg: str = "#ffffff",
    key: str | None = None,
) -> str:
    freq      = _build_frequencies(texts)
    colors    = _PALETTES.get(colormap, _PALETTES["Blues"])
    top_words = freq.most_common(60)

    word_list_json = json.dumps(top_words, ensure_ascii=False)
    colors_json    = json.dumps(colors,    ensure_ascii=False)

    html = _TEMPLATE.format(
        width          = width,
        height         = height,
        bg             = bg,
        word_list_json = word_list_json,
        colors_json    = colors_json,
    )

    components.html(html, height=height + 8, scrolling=False)
    return html