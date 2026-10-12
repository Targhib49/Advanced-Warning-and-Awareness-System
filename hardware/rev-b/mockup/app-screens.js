// The AWAS centre-screen app, drawn on a 1024 x 600 canvas.
//
// One renderer feeds both the 3D mockup (as a screen texture) and the paper's
// interface figure, so the two can't drift apart. The design follows §D-4 of
// the results: while the car moves the screen carries almost nothing to read.
// The band colour and the single lit segment give the level at a glance, and
// the words go out as audio. Detail appears only when the car is parked.
//
// The app speaks Indonesian by default; English is kept for an English paper.
//
// The live app (awas/display, served by the device) calls drawScreen with a
// fourth `live` argument carrying real numbers. Without it every screen draws
// its fixed sample values, which is what the mockup and the figure want.

export const W = 1024, H = 600;

const BG = "#0B0D0E", PANEL = "#15191C", TRACK = "#22292D", INK = "#E8EDEF", MUTED = "#8E9BA1";
const GREEN = "#38D39B", AMBER = "#E0A93C", ORANGE = "#F27A3E", SLATE = "#8A9AA2", RED = "#E5483A";
const DISPLAY = "'Barlow Semi Condensed', 'Arial Narrow', sans-serif";
const BODY = "'Barlow', system-ui, sans-serif";
const MONO = "'IBM Plex Mono', ui-monospace, monospace";

// Layout and state, independent of language.
export const SCREENS = {
  monitoring: { band: GREEN, level: 0, bothChannels: true },
  engaging:   { band: AMBER, level: 2, bothChannels: true, voice: true },
  gate:       { band: ORANGE, level: 3, bothChannels: true, note: true },
  occluded:   { band: SLATE, level: 2, bothChannels: false, capped: true, note: true },
  critical:   { flood: RED, bothChannels: true },
  parked:     { parked: true, band: GREEN },
  // Live-app screens the mockup does not show.
  tired:       { band: AMBER, level: 1, bothChannels: true },
  insist:      { band: ORANGE, level: 3, bothChannels: true },
  calibrating: { band: SLATE, level: -1, bothChannels: true, progress: true },
  noface:      { band: SLATE, level: -1, bothChannels: false, note: true },
};

// Every string on screen. Indonesian uses a decimal comma.
export const TEXT = {
  id: {
    channels: ["WAJAH + LAJUR", "LAJUR"],
    capped: "DIBATASI DI LEVEL 3",
    monitoring: { word: "SIAGA", line: "Tak ada yang perlu dibaca. Mata ke jalan.", foot: ["62 km/jam", "lajur aman", "beban 0,31"] },
    engaging: { word: "MENGANTUK", line: "Ada pertanyaan untuk Anda. Jawab dengan suara.", voice: "Mendengarkan", foot: ["58 km/jam", "lajur aman", "beban 0,28"] },
    gate: { word: "TIDAK AMAN", line: "Peringatan ditahan: lalu lintas padat.", note: "Saya akan bicara begitu jalan lebih lengang.", foot: ["41 km/jam", "padat merayap", "beban 0,86"] },
    occluded: { word: "HANYA LAJUR", line: "Kamera terhalang. Lajur tetap dipantau.", note: "Oleng 8 cm lebih dari biasanya (SDLP 0,21 m)", foot: ["54 km/jam", "lajur aman", "beban 0,33"] },
    critical: { word: "SEGERA MENEPI", line: "Microsleep terdeteksi", sub: "Rest area 3 km lagi, di sebelah kiri", foot: ["61 km/jam"] },
    parked: {
      header: "PARKIR · PERJALANAN SELESAI",
      stats: [["3:12", "mengemudi"], ["218 km", "jarak"], ["7", "pengingat"]],
      chart: "KANTUK SELAMA PERJALANAN", worst: "terparah 22:50–23:05",
      kss: "Seberapa mengantuk Anda?", scale: "1 sangat waspada · 9 melawan kantuk",
    },
    faceOnly: "WAJAH",
    tired: { word: "LELAH", line: "Mulai lelah. Sebentar lagi saya menyapa Anda.", foot: [] },
    insist: { word: "TIDAK AMAN", line: "Cari tempat aman untuk berhenti.", foot: [] },
    calibrating: { word: "KALIBRASI", line: "Mengemudi seperti biasa. Saya sedang mempelajari mata Anda.", foot: [] },
    noface: { word: "TAK TERLIHAT", line: "Kamera tidak melihat mata Anda.", note: "Periksa kamera dan posisi duduk.", foot: [] },
  },
  en: {
    channels: ["FACE + LANE", "LANE"],
    capped: "CAPPED AT LEVEL 3",
    monitoring: { word: "AWAKE", line: "Nothing to read. Eyes up.", foot: ["62 km/h", "lane ok", "workload 0.31"] },
    engaging: { word: "DROWSY", line: "Asked you a question. Answer out loud.", voice: "Listening", foot: ["58 km/h", "lane ok", "workload 0.28"] },
    gate: { word: "UNSAFE", line: "Holding back: heavy traffic.", note: "I'll speak as soon as the road eases.", foot: ["41 km/h", "stop-go", "workload 0.86"] },
    occluded: { word: "LANE ONLY", line: "Camera blocked. Still watching the lane.", note: "Weaving 8 cm more than your usual (SDLP 0.21 m)", foot: ["54 km/h", "lane ok", "workload 0.33"] },
    critical: { word: "PULL OVER", line: "Microsleep detected", sub: "Rest area in 3 km, on the left", foot: ["61 km/h"] },
    parked: {
      header: "PARKED · TRIP COMPLETE",
      stats: [["3:12", "driving"], ["218 km", "distance"], ["7", "nudges"]],
      chart: "DROWSINESS OVER THE TRIP", worst: "worst 22:50–23:05",
      kss: "How sleepy do you feel?", scale: "1 very alert · 9 fighting sleep",
    },
    faceOnly: "FACE",
    tired: { word: "TIRED", line: "Getting tired. I'll check in with you shortly.", foot: [] },
    insist: { word: "UNSAFE", line: "Find a safe place to stop.", foot: [] },
    calibrating: { word: "CALIBRATING", line: "Drive as usual. Learning your eyes.", foot: [] },
    noface: { word: "CAN'T SEE YOU", line: "The camera can't see your eyes.", note: "Check the camera and your seating position.", foot: [] },
  },
};

// Light-bar colour on the device for each screen.
export const LIGHT = { monitoring: GREEN, engaging: AMBER, gate: ORANGE, occluded: SLATE, critical: RED, parked: GREEN,
                       tired: AMBER, insist: ORANGE, calibrating: SLATE, noface: SLATE };

function rr(c, x, y, w, h, r) { c.beginPath(); c.roundRect(x, y, w, h, r); }
function t(c, s, x, y, font, color, align = "left", base = "alphabetic") {
  c.font = font; c.fillStyle = color; c.textAlign = align; c.textBaseline = base; c.fillText(s, x, y);
  return c.measureText(s).width;
}
// Shrink a headline until it fits the width available.
function fit(c, s, size, weight, family, maxW) {
  let px = size;
  for (; px > 60; px -= 4) { c.font = `${weight} ${px}px ${family}`; if (c.measureText(s).width <= maxW) break; }
  return `${weight} ${px}px ${family}`;
}

function statusRow(c, s, T, light, L) {
  const ink = light ? "rgba(255,255,255,0.86)" : MUTED;
  c.fillStyle = light ? "#FFFFFF" : s.bothChannels ? GREEN : SLATE;
  c.beginPath(); c.arc(50, 52, 8, 0, Math.PI * 2); c.fill();
  const label = L?.channel ?? T.channels[s.bothChannels ? 0 : 1];
  let x = 68 + t(c, label, 68, 61, `500 24px ${MONO}`, ink);
  if (s.capped) {
    x += 22;
    c.font = `500 18px ${MONO}`;
    const w = c.measureText(T.capped).width + 36;
    c.strokeStyle = AMBER; c.lineWidth = 2; rr(c, x, 34, w, 38, 19); c.stroke();
    t(c, T.capped, x + w / 2, 60, `500 18px ${MONO}`, AMBER, "center");
  }
  t(c, L?.clock ?? "23:41", W - 40, 61, `500 24px ${MONO}`, ink, "right");
}

function footer(c, foot, light) {
  if (light) { c.fillStyle = "rgba(0,0,0,0.16)"; c.fillRect(0, 516, W, H - 516); }
  else { c.fillStyle = TRACK; c.fillRect(40, 516, W - 80, 2); }
  let x = 40;
  for (const item of foot) x += t(c, item, x, 568, `500 24px ${MONO}`, light ? "rgba(255,255,255,0.9)" : MUTED) + 44;
}

function moving(c, s, T, S, L) {
  c.fillStyle = BG; c.fillRect(0, 0, W, H);
  c.fillStyle = s.band; c.fillRect(0, 0, W, 10);
  statusRow(c, s, T, false, L);
  if (s.progress) {
    // Calibration: one bar filling up instead of the five level segments.
    rr(c, 40, 104, W - 80, 18, 9); c.fillStyle = TRACK; c.fill();
    const p = Math.max(0, Math.min(1, L?.progress ?? 0.4));
    if (p > 0) { rr(c, 40, 104, Math.max(18, (W - 80) * p), 18, 9); c.fillStyle = s.band; c.fill(); }
  } else {
    // Five segments, one lit: the level at a glance.
    const segW = (W - 80 - 4 * 14) / 5;
    const lit = L?.level ?? s.level;
    for (let i = 0; i < 5; i++) {
      c.fillStyle = i === lit ? s.band : TRACK;
      rr(c, 40 + i * (segW + 14), 104, segW, 18, 9); c.fill();
    }
  }
  t(c, S.word, 34, 300, fit(c, S.word, 172, 700, DISPLAY, W - 72), s.band);
  t(c, S.line, 40, 364, fit(c, S.line, 36, 500, BODY, W - 80), INK);
  if (s.voice) {
    c.font = `600 28px ${BODY}`;
    const w = c.measureText(S.voice).width + 156;
    rr(c, 40, 404, w, 64, 32); c.fillStyle = PANEL; c.fill();
    [14, 28, 40, 24, 34, 18].forEach((h, i) => { c.fillStyle = s.band; rr(c, 70 + i * 13, 436 - h / 2, 7, h, 3.5); c.fill(); });
    t(c, S.voice, 158, 446, `600 28px ${BODY}`, INK);
  }
  if (s.note) {
    rr(c, 40, 404, W - 80, 78, 14); c.fillStyle = PANEL; c.fill();
    c.fillStyle = s.band; rr(c, 40, 404, 8, 78, 4); c.fill();
    t(c, S.note, 74, 454, fit(c, S.note, 30, 500, BODY, W - 140), INK);
  }
  footer(c, S.foot, false);
}

function critical(c, s, T, S, L) {
  c.fillStyle = s.flood; c.fillRect(0, 0, W, H);
  statusRow(c, s, T, true, L);
  t(c, S.word, W / 2, 316, fit(c, S.word, 200, 700, DISPLAY, W - 80), "#FFFFFF", "center");
  t(c, S.line, W / 2, 384, `600 38px ${BODY}`, "#FFFFFF", "center");
  t(c, S.sub, W / 2, 436, `500 30px ${BODY}`, "rgba(255,255,255,0.86)", "center");
  footer(c, S.foot, true);
}

function parked(c, T, L) {
  const S = { ...T.parked, ...(L?.text ?? {}) };
  c.fillStyle = BG; c.fillRect(0, 0, W, H);
  c.fillStyle = GREEN; c.fillRect(0, 0, W, 10);
  c.fillStyle = MUTED; c.beginPath(); c.arc(50, 52, 8, 0, Math.PI * 2); c.fill();
  t(c, S.header, 68, 61, `500 24px ${MONO}`, MUTED);
  t(c, L?.clock ?? "23:42", W - 40, 61, `500 24px ${MONO}`, MUTED, "right");
  // Trip numbers.
  S.stats.forEach(([v, k], i) => {
    const y = 160 + i * 96;
    t(c, v, 40, y, `700 64px ${DISPLAY}`, INK);
    t(c, k.toUpperCase(), 40, y + 30, `500 18px ${MONO}`, MUTED);
  });
  // Drowsiness over the trip, with the worst stretch marked.
  const x0 = 300, x1 = W - 40, y0 = 112, y1 = 360;
  rr(c, x0 - 20, y0 - 24, x1 - x0 + 40, y1 - y0 + 92, 14); c.fillStyle = PANEL; c.fill();
  t(c, S.chart, x0, y0 + 4, `500 18px ${MONO}`, MUTED);
  let pts = L?.chart ?? [0.12, 0.14, 0.13, 0.18, 0.2, 0.17, 0.22, 0.26, 0.3, 0.42, 0.55, 0.62, 0.58, 0.4, 0.3, 0.33, 0.28, 0.24];
  if (pts.length === 1) pts = [pts[0], pts[0]];
  const worst = L ? L.worst : [9, 13];
  const top = y0 + 30, gh = y1 - top;
  const X = i => x0 + (pts.length > 1 ? i / (pts.length - 1) : 0) * (x1 - x0), Y = v => y1 - v * gh;
  if (worst) { c.fillStyle = "rgba(224,169,60,0.16)"; c.fillRect(X(worst[0]) - 4, top, X(worst[1]) - X(worst[0]) + 8, gh); }
  c.strokeStyle = TRACK; c.lineWidth = 2;
  for (const v of [0.25, 0.5, 0.75]) { c.beginPath(); c.moveTo(x0, Y(v)); c.lineTo(x1, Y(v)); c.stroke(); }
  if (pts.length) {
    c.beginPath(); pts.forEach((v, i) => i ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v)));
    c.strokeStyle = GREEN; c.lineWidth = 4; c.lineJoin = "round"; c.stroke();
    c.lineTo(x1, y1); c.lineTo(x0, y1); c.closePath(); c.fillStyle = "rgba(56,211,155,0.10)"; c.fill();
  }
  t(c, L?.chartStart ?? "20:30", x0, y1 + 34, `500 18px ${MONO}`, MUTED);
  t(c, L?.chartEnd ?? "23:42", x1, y1 + 34, `500 18px ${MONO}`, MUTED, "right");
  if (worst && S.worst) t(c, S.worst, (X(worst[0]) + X(worst[1])) / 2, top + 14, `500 18px ${MONO}`, AMBER, "center");
  // KSS self-report: question on the left, nine buttons on the right.
  t(c, S.kss, 40, 494, fit(c, S.kss, 30, 600, BODY, 420), INK);
  t(c, S.scale, 40, 528, fit(c, S.scale, 20, 500, MONO, 430), MUTED);
  const chosen = L ? L.kss : 6;
  for (let k = 1; k <= 9; k++) {
    const bx = 490 + (k - 1) * 56, on = k === chosen;
    rr(c, bx, 462, 46, 92, 12); c.fillStyle = on ? AMBER : TRACK; c.fill();
    t(c, String(k), bx + 23, 520, `600 34px ${BODY}`, on ? BG : INK, "center");
  }
}

/** Where the KSS buttons sit on the parked screen, for hit-testing taps. */
export function kssButtonAt(x, y) {
  if (y < 462 || y > 554) return null;
  for (let k = 1; k <= 9; k++) { const bx = 490 + (k - 1) * 56; if (x >= bx && x <= bx + 46) return k; }
  return null;
}

/** Draw one screen at 1024 x 600 logical pixels (scale the context first).
 *  `live`, when given, replaces the sample values: { clock, channel, progress, level,
 *  text: {word, line, note, sub, foot, ...}, chart, worst, chartStart, chartEnd, kss }. */
export function drawScreen(c, key, lang = "id", live = null) {
  const s = SCREENS[key], T = TEXT[lang];
  const S = s && !s.parked ? { ...T[key], ...(live?.text ?? {}) } : null;
  c.save();
  if (!s) { c.fillStyle = "#000"; c.fillRect(0, 0, W, H); }
  else if (s.parked) parked(c, T, live);
  else if (s.flood) critical(c, s, T, S, live);
  else moving(c, s, T, S, live);
  c.restore();
}
