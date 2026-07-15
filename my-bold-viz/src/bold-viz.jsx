import { useState, useMemo, useEffect } from "react";

/* ============================================================================
   Reading Emotion in the Visual Brain — a plain-language walkthrough
   Studies: Bo et al. (2021) Cerebral Cortex (fMRI MVPA + LPP)
            Bo et al. (2022) NeuroImage (EEG timing + EEG–fMRI RSA)
   Real assets used from this repo:
     · /media/glm_convolution.png, /media/design_matrix.png  (pipeline figures)
     · /media/decoding_real.png  (group MVPA accuracy figure)
     · /media/rsa_real.png       (RSA onset-time summary)
     · REAL VVC decoding accuracies (20 subjects) inlined below
   Illustrative reconstructions (labelled): ERP waveform, generalization map,
   the LPP↔decoding scatter (no per-subject LPP is in this repo).
   ========================================================================== */

// ─── real data: VVC decoding accuracy, 20 subjects (this lab's SPM pipeline) ──
const VVC_PL = [57.7, 59.7, 61.2, 62.2, 60.1, 61.5, 54.0, 62.7, 67.0, 63.5, 64.6, 54.7, 60.3, 53.0, 58.9, 62.9, 55.3, 57.0, 56.4, 61.3];
const VVC_UP = [59.4, 61.2, 65.9, 68.8, 59.2, 59.3, 58.4, 65.4, 58.5, 74.2, 67.6, 57.3, 59.9, 57.5, 57.8, 66.8, 59.9, 61.6, 56.7, 65.1];
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;

// ─── seeded RNG for illustrative clouds ─────────────────────────────────────
function rng(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r) { const u = r() || 1e-9, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

const PL = "var(--pl)", NT = "var(--nt)", UP = "var(--up)";

// ─── layout primitives ──────────────────────────────────────────────────────
function Section({ id, kicker, n, title, children }) {
  return (
    <section id={id} className="sec">
      <div className="sec-head">
        {n != null && <span className="sec-num">{n}</span>}
        <div>{kicker && <div className="kicker">{kicker}</div>}<h2>{title}</h2></div>
      </div>
      {children}
    </section>
  );
}
function Callout({ children, tone = "accent" }) { return <div className={`callout ${tone}`}>{children}</div>; }
function Stat({ value, unit, label, color }) {
  return (
    <div className="stat">
      <div className="stat-val" style={color ? { color } : undefined}>{value}<span className="stat-unit">{unit}</span></div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
function Chip({ color, children }) {
  return <span className="chip"><span className="chip-dot" style={{ background: color }} />{children}</span>;
}
function Figure({ caption, children }) {
  return <figure className="fig"><div className="fig-body">{children}</div>{caption && <figcaption>{caption}</figcaption>}</figure>;
}
function Img({ src, alt, caption, real }) {
  return (
    <figure className="fig">
      <div className="fig-body img">
        {real && <span className="real-tag">actual research output</span>}
        <img src={src} alt={alt} loading="lazy" />
      </div>
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  );
}

// ============================================================================
// PIPELINE ROADMAP — the whole journey at a glance
// ============================================================================
const STEPS = [
  ["📷", "Show a picture", "pleasant, neutral or unpleasant"],
  ["🧲", "Scan the brain", "MRI blood flow + EEG waves"],
  ["🧹", "Clean the data", "remove movement & drift"],
  ["#️⃣", "One number per picture", "the response strength"],
  ["▦", "Build a pattern table", "pictures × brain pixels"],
  ["🎯", "Zoom into a region", "e.g. V1, the first stop"],
  ["🤖", "Teach a computer", "pleasant vs neutral?"],
  ["✅", "Check it's not luck", "cross-check & shuffle test"],
  ["📊", "Read the answer", "where, when & how emotion shows up"],
];
function PipelineMap() {
  return (
    <div className="roadmap">
      {STEPS.map(([ic, t, s], i) => (
        <div className="rm-item" key={i}>
          <div className="rm-card">
            <div className="rm-ic">{ic}</div>
            <div className="rm-n">Step {i + 1}</div>
            <div className="rm-t">{t}</div>
            <div className="rm-s">{s}</div>
          </div>
          {i < STEPS.length - 1 && <div className="rm-arrow">→</div>}
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// HRF — slow blood response
// ============================================================================
function HrfChart() {
  const W = 720, H = 240, P = { t: 24, r: 24, b: 44, l: 48 };
  const iw = W - P.l - P.r, ih = H - P.t - P.b;
  const pts = [];
  for (let s = 0; s <= 16; s += 0.25) {
    const y = Math.pow(s, 6) * Math.exp(-s) / 700 - 0.35 * Math.pow(s / 3, 6) * Math.exp(-s / 3) / 30;
    pts.push([s, y]);
  }
  const ymax = Math.max(...pts.map(p => p[1])), ymin = Math.min(...pts.map(p => p[1]));
  const x = s => P.l + (s / 16) * iw, y = v => P.t + ih - ((v - ymin) / (ymax - ymin)) * ih;
  const path = pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(" ");
  const peakS = pts.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Blood-flow response peaks about six seconds after a picture.">
      <line x1={P.l} y1={y(0)} x2={W - P.r} y2={y(0)} className="axis" />
      {[0, 4, 8, 12, 16].map(s => (
        <g key={s}><line x1={x(s)} y1={P.t} x2={x(s)} y2={P.t + ih} className="grid" /><text x={x(s)} y={H - 22} className="tick" textAnchor="middle">{s}s</text></g>
      ))}
      <line x1={x(0)} y1={P.t} x2={x(0)} y2={P.t + ih} className="stim" />
      <text x={x(0) + 6} y={P.t + 12} className="anno">picture shown</text>
      <path d={path} fill="none" stroke={NT} strokeWidth="2.5" />
      <circle cx={x(peakS)} cy={y(ymax)} r="5" fill={NT} />
      <text x={x(peakS)} y={y(ymax) - 12} className="anno" textAnchor="middle">peak ≈ 6s later</text>
      <text x={W / 2} y={H - 4} className="axis-label" textAnchor="middle">time after the picture appears</text>
    </svg>
  );
}

// ============================================================================
// BETA SWATCHES — from a wiggly signal to one number per picture
// ============================================================================
function BetaSwatches() {
  const W = 720, H = 250;
  const conds = useMemo(() => {
    const r = rng(5), out = [];
    for (let i = 0; i < 15; i++) {
      const c = i % 3 === 0 ? "pl" : i % 3 === 1 ? "nt" : "up";
      out.push({ c, v: (c === "up" ? 2.4 : c === "pl" ? 1.9 : 0.9) + gauss(r) * 0.4 });
    }
    return out;
  }, []);
  // little wiggly line
  const lw = [];
  const rr = rng(9);
  for (let i = 0; i <= 120; i++) {
    let v = 0;
    for (let k = 0; k < 15; k++) v += Math.exp(-Math.pow((i - (8 + k * 7)) / 3, 2)) * conds[k].v;
    lw.push([i, v + gauss(rr) * 0.15]);
  }
  const lmax = Math.max(...lw.map(p => p[1]));
  const x = i => 40 + (i / 120) * (W - 60), y = v => 90 - (v / lmax) * 66;
  const path = lw.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(" ");
  const col = c => (c === "pl" ? PL : c === "nt" ? NT : UP);
  const sw = (W - 60) / 15;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="A wiggly brain signal is summarised into one response number for each picture.">
      <text x={40} y={20} className="anno">brain signal over time (many pictures overlap)</text>
      <path d={path} fill="none" stroke="var(--muted)" strokeWidth="1.6" />
      <line x1={40} y1={100} x2={W - 20} y2={100} className="grid" />
      <text x={W / 2} y={132} className="anno" textAnchor="middle" style={{ fill: "var(--accent)" }}>↓ summarise each picture into ONE number ↓</text>
      {conds.map((d, i) => {
        const cx = 40 + i * sw + sw / 2, h = 8 + (d.v / 2.8) * 58;
        return (
          <g key={i}>
            <rect x={cx - sw * 0.34} y={228 - h} width={sw * 0.68} height={h} rx="3" fill={col(d.c)} opacity="0.85" />
            <text x={cx} y={228 - h - 4} className="swatch-v" textAnchor="middle">{d.v.toFixed(1)}</text>
          </g>
        );
      })}
      <text x={40} y={244} className="tick">picture 1</text>
      <text x={W - 20} y={244} className="tick" textAnchor="end">picture 15…</text>
    </svg>
  );
}

// ============================================================================
// PATTERN TABLE — pictures (rows) × brain pixels (columns)
// ============================================================================
function PatternTable() {
  const rows = 18, cols = 8, W = 420, cell = 22, top = 34, left = 78;
  const data = useMemo(() => {
    const r = rng(3), out = [];
    for (let i = 0; i < rows; i++) {
      const g = i < 6 ? "pl" : i < 12 ? "nt" : "up";
      const base = g === "pl" ? [0.8, 0.2, 0.85, 0.25, 0.7, 0.3, 0.6, 0.35]
        : g === "nt" ? [0.45, 0.5, 0.42, 0.55, 0.48, 0.5, 0.46, 0.52]
        : [0.9, 0.15, 0.95, 0.1, 0.8, 0.2, 0.7, 0.25];
      out.push({ g, v: base.map(b => Math.max(0, Math.min(1, b + gauss(r) * 0.12))) });
    }
    return out;
  }, []);
  const col = (g, v) => {
    const c = g === "pl" ? [18, 160, 110] : g === "nt" ? [42, 120, 214] : [227, 73, 72];
    const bg = 245;
    return `rgb(${c.map(x => Math.round(bg + (x - bg) * v)).join(",")})`;
  };
  const H = top + rows * cell + 24;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="A table: each row is a picture, each column a brain pixel; colour shows the response.">
      {["pleasant", "neutral", "unpleasant"].map((lbl, k) => (
        <text key={lbl} x={left - 8} y={top + (k * 6 + 3) * cell + 4} className="tick"
              textAnchor="end" style={{ fill: k === 0 ? PL : k === 1 ? NT : UP, fontWeight: 700 }}>{lbl}</text>
      ))}
      <text x={left + cols * cell / 2} y={20} className="anno" textAnchor="middle">brain pixels (voxels) →</text>
      {data.map((row, i) => row.v.map((v, j) => (
        <rect key={`${i}-${j}`} x={left + j * cell} y={top + i * cell} width={cell - 1.5} height={cell - 1.5}
              rx="2" fill={col(row.g, v)} />
      )))}
      <text x={left - 8} y={top - 8} className="tick" textAnchor="end">picture ↓</text>
    </svg>
  );
}

// ============================================================================
// BRAIN REGION — zoom into one visual area
// ============================================================================
function BrainRegion() {
  return (
    <svg viewBox="0 0 720 240" className="chart" role="img" aria-label="A side view of the brain with the visual cortex at the back highlighted.">
      {/* simple brain silhouette (side view, facing left) */}
      <path d="M120,140 C110,70 190,30 300,32 C430,34 560,55 600,110 C625,145 600,190 540,200 C560,205 545,222 520,214 C500,232 470,222 470,206 C420,214 360,214 300,208 C230,214 150,205 130,175 C118,168 112,152 120,140 Z"
            fill="var(--surface)" stroke="var(--axis)" strokeWidth="2" />
      {/* occipital / visual cortex blob at the back (right side) */}
      <path d="M540,90 C585,95 605,130 585,168 C560,200 505,198 495,160 C488,128 505,95 540,90 Z"
            fill="var(--accent)" opacity="0.85" />
      <text x="540" y="150" className="node-t" textAnchor="middle" style={{ fill: "#fff" }}>visual</text>
      <text x="540" y="166" className="node-t" textAnchor="middle" style={{ fill: "#fff" }}>cortex</text>
      <line x1="500" y1="70" x2="470" y2="40" stroke="var(--accent)" strokeWidth="1.5" />
      <text x="466" y="34" className="anno" textAnchor="end" style={{ fill: "var(--accent)" }}>where the eyes' signal first arrives</text>
      <text x="230" y="120" className="node-s" textAnchor="middle">front of brain</text>
      {/* eye */}
      <circle cx="95" cy="150" r="16" fill="var(--surface)" stroke="var(--nt)" strokeWidth="2" />
      <circle cx="90" cy="150" r="6" fill="var(--nt)" />
      <text x="95" y="188" className="tick" textAnchor="middle">eye</text>
      <path d="M111,150 C300,250 430,220 500,175" fill="none" stroke="var(--nt)" strokeWidth="1.5" strokeDasharray="4 3" />
    </svg>
  );
}

// ============================================================================
// DECISION BOUNDARY — teaching the classifier
// ============================================================================
function DecisionBoundary() {
  const W = 420, H = 320, P = 34;
  const pts = useMemo(() => {
    const r = rng(11), out = [];
    for (let i = 0; i < 22; i++) out.push({ c: "pl", x: 0.62 + gauss(r) * 0.13, y: 0.62 + gauss(r) * 0.13 });
    for (let i = 0; i < 22; i++) out.push({ c: "nt", x: 0.38 + gauss(r) * 0.13, y: 0.38 + gauss(r) * 0.13 });
    return out;
  }, []);
  const sx = v => P + v * (W - 2 * P), sy = v => (H - P) - v * (H - 2 * P);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Two clouds of dots separated by a diagonal line — the computer's decision rule.">
      <line x1={P} y1={H - P} x2={W - P} y2={H - P} className="axis" />
      <line x1={P} y1={P} x2={P} y2={H - P} className="axis" />
      <text x={W / 2} y={H - 6} className="axis-label" textAnchor="middle">activity in brain pixel A</text>
      <text x={14} y={H / 2} className="axis-label" textAnchor="middle" transform={`rotate(-90 14 ${H / 2})`}>activity in brain pixel B</text>
      {/* boundary line: x + y = 1 */}
      <line x1={sx(0.05)} y1={sy(0.95)} x2={sx(0.95)} y2={sy(0.05)} stroke="var(--accent)" strokeWidth="2.5" />
      <text x={sx(0.8)} y={sy(0.82)} className="anno" style={{ fill: "var(--accent)" }} textAnchor="middle">best dividing line</text>
      {pts.map((p, i) => (
        <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r="5" fill={p.c === "pl" ? PL : NT}
                opacity="0.82" stroke="var(--surface)" strokeWidth="1" />
      ))}
      <g transform={`translate(${P + 6},${P})`}>
        <circle cx="6" cy="0" r="5" fill={PL} /><text x="16" y="4" className="tick">pleasant</text>
        <circle cx="6" cy="18" r="5" fill={NT} /><text x="16" y="22" className="tick">neutral</text>
      </g>
    </svg>
  );
}

// ============================================================================
// CROSS-VALIDATION — train on most, test on the rest, repeat
// ============================================================================
function CrossVal() {
  const folds = 5, W = 560, bw = 84, bh = 26, gap = 8, top = 34, left = 70;
  return (
    <svg viewBox={`0 0 ${W} ${top + folds * (bh + gap) + 20}`} className="chart" role="img"
         aria-label="Five rounds: in each, four blocks train the computer and one tests it, rotating which block is the test.">
      <text x={left} y={20} className="anno">the pictures are split into 5 blocks…</text>
      {Array.from({ length: folds }).map((_, f) => (
        <g key={f}>
          <text x={left - 10} y={top + f * (bh + gap) + bh / 2 + 4} className="tick" textAnchor="end">round {f + 1}</text>
          {Array.from({ length: folds }).map((_, b) => {
            const test = b === f;
            return (
              <g key={b}>
                <rect x={left + b * (bw + 4)} y={top + f * (bh + gap)} width={bw} height={bh} rx="4"
                      fill={test ? "var(--accent)" : "var(--nt)"} opacity={test ? 0.9 : 0.32} />
                <text x={left + b * (bw + 4) + bw / 2} y={top + f * (bh + gap) + bh / 2 + 4}
                      className="cv-lbl" textAnchor="middle" style={{ fill: test ? "#fff" : "var(--nt)" }}>
                  {test ? "TEST" : "train"}
                </text>
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

// ============================================================================
// PERMUTATION — is the score better than dumb luck?
// ============================================================================
function PermutationHist() {
  const W = 720, H = 260, P = { t: 24, r: 20, b: 44, l: 20 };
  const iw = W - P.l - P.r, ih = H - P.t - P.b;
  const bins = 41, lo = 42, hi = 68;
  const counts = useMemo(() => {
    const r = rng(21), c = Array(bins).fill(0);
    for (let i = 0; i < 4000; i++) {
      const v = 50 + gauss(r) * 2.2;
      const b = Math.floor(((v - lo) / (hi - lo)) * bins);
      if (b >= 0 && b < bins) c[b]++;
    }
    return c;
  }, []);
  const cmax = Math.max(...counts);
  const x = v => P.l + ((v - lo) / (hi - lo)) * iw;
  const bw = iw / bins;
  const obs = 62;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img"
         aria-label="A bell curve of chance scores around 50 percent; the real score sits far out to the right at 62 percent.">
      <line x1={P.l} y1={P.t + ih} x2={W - P.r} y2={P.t + ih} className="axis" />
      {counts.map((c, i) => {
        const v = lo + (i + 0.5) * (hi - lo) / bins;
        const h = (c / cmax) * ih;
        return <rect key={i} x={x(v) - bw / 2} y={P.t + ih - h} width={bw - 1} height={h} fill="var(--nt)" opacity="0.3" />;
      })}
      <text x={x(50)} y={P.t + 12} className="anno" textAnchor="middle" style={{ fill: "var(--muted)" }}>scores when labels are shuffled (pure luck)</text>
      <line x1={x(50)} y1={P.t} x2={x(50)} y2={P.t + ih} className="ref chance" />
      <line x1={x(obs)} y1={P.t} x2={x(obs)} y2={P.t + ih} stroke={UP} strokeWidth="2.5" />
      <circle cx={x(obs)} cy={P.t + 6} r="5" fill={UP} />
      <text x={x(obs)} y={P.t - 2} className="anno" textAnchor="middle" style={{ fill: UP }}>real score ≈ 62%</text>
      {[45, 50, 55, 60, 65].map(v => <text key={v} x={x(v)} y={H - 22} className="tick" textAnchor="middle">{v}%</text>)}
      <text x={W / 2} y={H - 4} className="axis-label" textAnchor="middle">decoding accuracy</text>
    </svg>
  );
}

// ============================================================================
// DECODING per region (matches real fig1; approx group values)
// ============================================================================
const ROI_ACC = [
  { r: "V1v", a: 63 }, { r: "V1d", a: 64 }, { r: "V2v", a: 61 }, { r: "V2d", a: 62 },
  { r: "V3v", a: 62 }, { r: "V3d", a: 65 }, { r: "hV4", a: 63 }, { r: "VO1", a: 63 },
  { r: "VO2", a: 64 }, { r: "PHC1", a: 64 }, { r: "PHC2", a: 62 }, { r: "hMT", a: 68 },
  { r: "LO1", a: 67 }, { r: "LO2", a: 66 }, { r: "V3a", a: 63 }, { r: "V3b", a: 65 }, { r: "IPS", a: 64 },
];
function DecodingChart() {
  const [hover, setHover] = useState(null);
  const W = 720, H = 320, P = { t: 20, r: 16, b: 64, l: 44 };
  const iw = W - P.l - P.r, ih = H - P.t - P.b, lo = 45, hi = 72;
  const n = ROI_ACC.length, bw = (iw / n) * 0.62, step = iw / n;
  const y = v => P.t + ih - ((v - lo) / (hi - lo)) * ih, x = i => P.l + i * step + step / 2;
  return (
    <div style={{ position: "relative" }}>
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Every visual region decodes above the 54% significance line.">
        {[50, 55, 60, 65, 70].map(v => (
          <g key={v}><line x1={P.l} y1={y(v)} x2={W - P.r} y2={y(v)} className="grid" /><text x={P.l - 8} y={y(v) + 4} className="tick" textAnchor="end">{v}%</text></g>
        ))}
        <line x1={P.l} y1={y(50)} x2={W - P.r} y2={y(50)} className="ref chance" />
        <text x={W - P.r} y={y(50) - 6} className="ref-lbl" textAnchor="end">chance = 50%</text>
        <line x1={P.l} y1={y(54)} x2={W - P.r} y2={y(54)} className="ref sig" />
        <text x={W - P.r} y={y(54) + 14} className="ref-lbl sig" textAnchor="end">significant &gt; 54%</text>
        {ROI_ACC.map((d, i) => {
          const active = hover === null || hover === i;
          return (
            <g key={d.r} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={x(i) - bw / 2} y={y(d.a)} width={bw} height={P.t + ih - y(d.a)} rx="3" className="bar"
                    opacity={active ? 1 : 0.35} style={{ fill: d.r.startsWith("V1") ? "var(--accent)" : "var(--seq)" }} />
              <text x={x(i)} y={y(d.a) - 5} className="bar-val" textAnchor="middle" opacity={hover === i ? 1 : 0}>{d.a}%</text>
              <text x={x(i)} y={P.t + ih + 14} className="tick roi" textAnchor="end" transform={`rotate(-45 ${x(i)} ${P.t + ih + 14})`}>{d.r}</text>
            </g>
          );
        })}
      </svg>
      <div className="chart-note">
        <Chip color="var(--accent)">V1 — the brain's very first visual stop</Chip>
        <Chip color="var(--seq)">other visual regions</Chip>
      </div>
    </div>
  );
}

// ============================================================================
// VVC DOTS — REAL per-subject accuracies
// ============================================================================
function VvcDots() {
  const W = 720, H = 210, P = { t: 40, r: 30, b: 40, l: 120 };
  const iw = W - P.l - P.r, lo = 48, hi = 76;
  const x = v => P.l + ((v - lo) / (hi - lo)) * iw;
  const rowY = { pl: P.t + 26, up: P.t + 86 };
  const row = (arr, key, color, label) => (
    <g>
      <text x={P.l - 12} y={rowY[key] + 4} className="tick" textAnchor="end" style={{ fill: color, fontWeight: 700 }}>{label}</text>
      <line x1={P.l} y1={rowY[key]} x2={W - P.r} y2={rowY[key]} className="grid" />
      {arr.map((v, i) => <circle key={i} cx={x(v)} cy={rowY[key] + (i % 2 ? -6 : 6)} r="5" fill={color} opacity="0.7" stroke="var(--surface)" strokeWidth="1" />)}
      <line x1={x(mean(arr))} y1={rowY[key] - 16} x2={x(mean(arr))} y2={rowY[key] + 16} stroke={color} strokeWidth="2.5" />
      <text x={x(mean(arr))} y={rowY[key] - 20} className="anno" textAnchor="middle" style={{ fill: color }}>avg {mean(arr).toFixed(1)}%</text>
    </g>
  );
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Every one of 20 subjects decodes above chance in ventral visual cortex.">
      <line x1={x(50)} y1={P.t + 4} x2={x(50)} y2={H - P.b + 4} className="ref chance" />
      <text x={x(50)} y={P.t - 6} className="ref-lbl" textAnchor="middle">chance 50%</text>
      {row(VVC_PL, "pl", PL, "pleasant")}
      {row(VVC_UP, "up", UP, "unpleasant")}
      {[50, 55, 60, 65, 70, 75].map(v => <text key={v} x={x(v)} y={H - 16} className="tick" textAnchor="middle">{v}%</text>)}
      <text x={W / 2} y={H - 2} className="axis-label" textAnchor="middle">decoding accuracy in ventral visual cortex (each dot = one person)</text>
    </svg>
  );
}

// ============================================================================
// TIMELINE — when emotion appears
// ============================================================================
function TimelineChart() {
  const W = 720, H = 210, P = { t: 30, r: 24, b: 40, l: 24 };
  const iw = W - P.l - P.r, T = 2000, x = ms => P.l + (ms / T) * iw, axisY = 150;
  const marks = [
    { ms: 90, label: "visual cortex registers the scene", color: "var(--seq)", up: true },
    { ms: 100, label: "object cortex sees it", color: "var(--seq2)", up: false },
    { ms: 200, label: "pleasant becomes readable", color: PL, up: true },
    { ms: 260, label: "unpleasant becomes readable", color: UP, up: false },
  ];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Perception at 90-100ms, emotion at 200-260ms, sustained to 2 seconds.">
      <rect x={x(200)} y={axisY - 10} width={x(2000) - x(200)} height="20" rx="10" className="sustain" />
      <text x={(x(200) + x(2000)) / 2} y={axisY + 34} className="anno" textAnchor="middle">emotion pattern stays switched on ≈ 2 whole seconds</text>
      <line x1={P.l} y1={axisY} x2={W - P.r} y2={axisY} className="axis" />
      {[0, 500, 1000, 1500, 2000].map(ms => (
        <g key={ms}><line x1={x(ms)} y1={axisY - 4} x2={x(ms)} y2={axisY + 4} className="axis" /><text x={x(ms)} y={axisY + 20} className="tick" textAnchor="middle">{ms === 0 ? "picture on" : `${ms} ms`}</text></g>
      ))}
      {marks.map(m => {
        const ly = m.up ? axisY - 34 : axisY + 10, ty = m.up ? axisY - 44 : axisY + 44;
        return (
          <g key={m.ms}>
            <line x1={x(m.ms)} y1={m.up ? ly : axisY} x2={x(m.ms)} y2={m.up ? axisY : ly + 24} stroke={m.color} strokeWidth="1.5" strokeDasharray="3 3" />
            <circle cx={x(m.ms)} cy={axisY} r="5" fill={m.color} />
            <text x={x(m.ms)} y={ty} className="anno" textAnchor="middle" style={{ fill: m.color }}>{m.ms} ms</text>
            <text x={x(m.ms)} y={m.up ? ty - 13 : ty + 13} className="tl-lbl" textAnchor="middle">{m.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ============================================================================
// GENERALIZATION MAP (illustrative)
// ============================================================================
function GeneralizationMap() {
  const grid = 26, W = 300, cell = W / grid;
  const cells = useMemo(() => {
    const out = [];
    for (let ti = 0; ti < grid; ti++) for (let tj = 0; tj < grid; tj++) {
      const a = (ti / grid) * 2000, b = (tj / grid) * 2000;
      const onset = Math.min(a, b) > 180 ? 1 : 0;
      const diag = Math.exp(-Math.pow((a - b) / 520, 2));
      out.push({ ti, tj, v: onset * (0.55 + 0.45 * diag) });
    }
    return out;
  }, []);
  const col = v => {
    const stops = [[205, 226, 251], [110, 167, 236], [57, 135, 229], [28, 92, 171], [13, 54, 107]];
    if (v <= 0) return "var(--grid)";
    const t = Math.min(1, v) * (stops.length - 1), i = Math.floor(t), f = t - i;
    const a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
    return `rgb(${a.map((c, k) => Math.round(c + (b[k] - c) * f)).join(",")})`;
  };
  const pad = 34;
  return (
    <svg viewBox={`0 0 ${W + pad} ${W + pad}`} className="chart" role="img" aria-label="A broad bright square after 200ms shows the emotion pattern is stable.">
      <g transform={`translate(${pad},4)`}>
        {cells.map(c => <rect key={`${c.ti}-${c.tj}`} x={c.tj * cell} y={c.ti * cell} width={cell + 0.5} height={cell + 0.5} fill={col(c.v)} />)}
        <line x1="0" y1="0" x2={W} y2={W} stroke="rgba(255,255,255,.5)" strokeWidth="1" strokeDasharray="2 3" />
        <text x={W / 2} y={W + 24} className="tick" textAnchor="middle">test time  →  2000 ms</text>
      </g>
      <text x="14" y={W / 2} className="tick" textAnchor="middle" transform={`rotate(-90 14 ${W / 2})`}>train time  →  2000 ms</text>
    </svg>
  );
}

// ============================================================================
// LPP waveform (illustrative; real relative amplitudes)
// ============================================================================
function LppWaveform() {
  const W = 720, H = 290, P = { t: 20, r: 20, b: 46, l: 52 };
  const iw = W - P.l - P.r, ih = H - P.t - P.b, t0 = -200, t1 = 2000;
  const x = ms => P.l + ((ms - t0) / (t1 - t0)) * iw;
  const ylo = -1.5, yhi = 4, y = v => P.t + ih - ((v - ylo) / (yhi - ylo)) * ih;
  const wave = peak => {
    const pts = [];
    for (let ms = t0; ms <= t1; ms += 20) {
      let v = 0.9 * Math.exp(-Math.pow((ms - 120) / 60, 2)) - 0.7 * Math.exp(-Math.pow((ms - 200) / 55, 2));
      const lpp = Math.exp(-Math.pow((ms - 520) / 260, 2)) + 0.5 * Math.exp(-Math.pow((ms - 1100) / 700, 2));
      v += peak * 1.35 * (ms > 250 ? lpp : lpp * Math.max(0, (ms - 150) / 100));
      pts.push([ms, v]);
    }
    return pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(" ");
  };
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Emotional pictures drive a larger slow LPP wave than neutral from 300ms.">
      <rect x={x(300)} y={P.t} width={x(800) - x(300)} height={ih} className="lpp-win" />
      <text x={(x(300) + x(800)) / 2} y={P.t + 12} className="anno" textAnchor="middle">LPP window</text>
      {[-1, 0, 1, 2, 3, 4].map(v => (<g key={v}><line x1={P.l} y1={y(v)} x2={W - P.r} y2={y(v)} className="grid" /><text x={P.l - 8} y={y(v) + 4} className="tick" textAnchor="end">{v}</text></g>))}
      <line x1={x(0)} y1={P.t} x2={x(0)} y2={P.t + ih} className="stim" />
      {[0, 500, 1000, 1500, 2000].map(ms => <text key={ms} x={x(ms)} y={H - 24} className="tick" textAnchor="middle">{ms}</text>)}
      <path d={wave(0.78)} fill="none" stroke={NT} strokeWidth="2.2" />
      <path d={wave(1.97)} fill="none" stroke={PL} strokeWidth="2.2" />
      <path d={wave(2.26)} fill="none" stroke={UP} strokeWidth="2.2" />
      <g transform={`translate(${x(1250)},${P.t + 10})`}>
        <circle cx="0" cy="0" r="4" fill={UP} /><text x="10" y="4" className="tick">unpleasant 2.26 µV</text>
        <circle cx="0" cy="18" r="4" fill={PL} /><text x="10" y="22" className="tick">pleasant 1.97 µV</text>
        <circle cx="0" cy="36" r="4" fill={NT} /><text x="10" y="40" className="tick">neutral 0.78 µV</text>
      </g>
      <text x={W / 2} y={H - 4} className="axis-label" textAnchor="middle">time after picture (ms)</text>
      <text x={16} y={P.t + ih / 2} className="axis-label" textAnchor="middle" transform={`rotate(-90 16 ${P.t + ih / 2})`}>brain wave (µV)</text>
    </svg>
  );
}

// ============================================================================
// LPP scatter (illustrative cloud; R & p real)
// ============================================================================
function LppScatter({ R, color, label, seed }) {
  const W = 340, H = 300, P = { t: 18, r: 16, b: 42, l: 42 };
  const iw = W - P.l - P.r, ih = H - P.t - P.b;
  const pts = useMemo(() => {
    const r = rng(seed), out = [];
    for (let i = 0; i < 20; i++) { const zx = gauss(r), zy = R * zx + Math.sqrt(1 - R * R) * gauss(r); out.push([2 + zx * 1.4, 60 + zy * 4.5]); }
    return out;
  }, [R, seed]);
  const xlo = -1.5, xhi = 6, ylo = 48, yhi = 74;
  const x = v => P.l + ((v - xlo) / (xhi - xlo)) * iw, y = v => P.t + ih - ((v - ylo) / (yhi - ylo)) * ih;
  const mx = pts.reduce((s, p) => s + p[0], 0) / pts.length, my = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  const b1 = pts.reduce((s, p) => s + (p[0] - mx) * (p[1] - my), 0) / pts.reduce((s, p) => s + (p[0] - mx) ** 2, 0);
  const b0 = my - b1 * mx;
  return (
    <figure className="scatter">
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={`${label}: decoding rises with LPP, R = ${R}.`}>
        {[50, 55, 60, 65, 70].map(v => (<g key={v}><line x1={P.l} y1={y(v)} x2={W - P.r} y2={y(v)} className="grid" /><text x={P.l - 6} y={y(v) + 4} className="tick" textAnchor="end">{v}</text></g>))}
        {[0, 2, 4].map(v => <text key={v} x={x(v)} y={H - 24} className="tick" textAnchor="middle">{v}</text>)}
        <line x1={x(xlo)} y1={y(b0 + b1 * xlo)} x2={x(xhi)} y2={y(b0 + b1 * xhi)} stroke={color} strokeWidth="2" opacity="0.85" />
        {pts.map((p, i) => <circle key={i} cx={x(p[0])} cy={y(p[1])} r="4.5" fill={color} opacity="0.8" stroke="var(--surface)" strokeWidth="1" />)}
        <text x={W - P.r} y={P.t + 14} className="scatter-R" textAnchor="end" style={{ fill: color }}>R = {R}</text>
        <text x={W / 2} y={H - 4} className="axis-label" textAnchor="middle">LPP size (µV)</text>
        <text x={14} y={P.t + ih / 2} className="axis-label" textAnchor="middle" transform={`rotate(-90 14 ${P.t + ih / 2})`}>decoding accuracy (%)</text>
      </svg>
      <figcaption>{label}</figcaption>
    </figure>
  );
}

// ============================================================================
// Reentry diagram
// ============================================================================
function ReentryDiagram() {
  return (
    <svg viewBox="0 0 720 200" className="chart" role="img" aria-label="Deep emotion centres feed back to visual cortex, sharpening the emotional pattern.">
      <defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="var(--accent)" /></marker></defs>
      <rect x="20" y="70" width="130" height="60" rx="10" className="node in" />
      <text x="85" y="96" className="node-t" textAnchor="middle">picture</text>
      <text x="85" y="114" className="node-s" textAnchor="middle">hits the eyes</text>
      <rect x="290" y="70" width="150" height="60" rx="10" className="node vis" />
      <text x="365" y="96" className="node-t" textAnchor="middle">visual cortex</text>
      <text x="365" y="114" className="node-s" textAnchor="middle">V1 · ventral · dorsal</text>
      <rect x="560" y="20" width="140" height="56" rx="10" className="node amy" />
      <text x="630" y="44" className="node-t" textAnchor="middle">amygdala</text>
      <text x="630" y="61" className="node-s" textAnchor="middle">emotion center</text>
      <rect x="560" y="124" width="140" height="56" rx="10" className="node front" />
      <text x="630" y="148" className="node-t" textAnchor="middle">frontal cortex</text>
      <text x="630" y="165" className="node-s" textAnchor="middle">IFG · VLPFC</text>
      <line x1="150" y1="100" x2="285" y2="100" stroke="var(--accent)" strokeWidth="2" markerEnd="url(#arrow)" />
      <path d="M560,48 Q470,50 442,82" fill="none" stroke="var(--up)" strokeWidth="2" strokeDasharray="5 3" markerEnd="url(#arrow)" />
      <path d="M560,152 Q470,150 442,118" fill="none" stroke="var(--pl)" strokeWidth="2" strokeDasharray="5 3" markerEnd="url(#arrow)" />
      <text x="500" y="34" className="edge-lbl" style={{ fill: "var(--up)" }}>feedback (R = 0.66)</text>
      <text x="500" y="182" className="edge-lbl" style={{ fill: "var(--pl)" }}>feedback</text>
    </svg>
  );
}

// ============================================================================
// NAV
// ============================================================================
const NAV = [
  ["q", "The question"], ["exp", "The experiment"], ["map", "The whole journey"],
  ["signal", "The slow signal"], ["beta", "One number per picture"], ["table", "The pattern table"],
  ["region", "Zoom into a region"], ["decode", "Teach the computer"], ["trust", "Is it real?"],
  ["results", "What we found"], ["when", "When emotion appears"], ["rsa", "Matching two views"],
  ["lpp", "The feedback wave"], ["takeaway", "The takeaway"],
];
function Nav({ active }) {
  return (
    <nav className="nav">
      {NAV.map(([id, label]) => (
        <a key={id} href={`#${id}`} className={active === id ? "on" : ""}><span className="nav-dot" /><span className="nav-lbl">{label}</span></a>
      ))}
    </nav>
  );
}

// ============================================================================
// APP
// ============================================================================
export default function App() {
  const [active, setActive] = useState("q");
  useEffect(() => {
    const obs = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) setActive(e.target.id); }), { rootMargin: "-45% 0px -45% 0px" });
    NAV.forEach(([id]) => { const el = document.getElementById(id); if (el) obs.observe(el); });
    return () => obs.disconnect();
  }, []);

  return (
    <div className="viz-root">
      <Nav active={active} />

      <header className="hero">
        <div className="hero-in">
          <div className="eyebrow">Visual neuroscience · explained step by step</div>
          <h1>Your eyes see a picture.<br /><em>Can we read the feeling from your brain?</em></h1>
          <p className="lede">
            When you look at a scene — a smiling couple, a plate of food, a car crash — the very first
            patch of your brain to handle it isn't the "emotion" part. It's plain visual cortex, which
            just processes shapes and edges. Three studies asked:
            <strong> is the emotion already written into that early visual signal?</strong> Below is the
            whole journey, one plain step at a time — from a photo on a screen to the answer.
          </p>
          <div className="hero-tags"><Chip color={PL}>Pleasant</Chip><Chip color={NT}>Neutral</Chip><Chip color={UP}>Unpleasant</Chip></div>
        </div>
      </header>

      <main>
        {/* 1 */}
        <Section id="q" n="1" kicker="The big idea" title="Feelings might live in the visual system itself">
          <p>
            For a long time, scientists pictured early visual cortex as a neutral camera: it captures
            <em> what</em> you see and passes it on to deeper regions that decide how you <em>feel</em>.
            Newer work suggests the camera isn't neutral — the emotional meaning of a scene may leave a
            fingerprint in the earliest visual areas, because emotion centres deeper in the brain reach
            back and nudge them.
          </p>
          <Callout>
            <strong>In plain terms:</strong> instead of asking "how <em>active</em> is the visual brain,"
            these studies ask "does the <em>pattern</em> of activity look different for a happy scene
            versus a frightening one?" — and whether a computer can tell them apart.
          </Callout>
        </Section>

        {/* 2 */}
        <Section id="exp" n="2" kicker="The setup" title="20 people, 60 pictures, one brain scanner">
          <p>
            Twenty volunteers lay in an MRI scanner while also wearing an EEG cap that records electrical
            brain waves. They viewed 60 standardised photos — 20 pleasant, 20 neutral, 20 unpleasant —
            repeated across five runs, each picture on screen for 3 seconds. No task, no buttons: just look.
          </p>
          <div className="stat-row">
            <Stat value="20" unit="" label="people scanned" />
            <Stat value="60" unit="" label="pictures per run" />
            <Stat value="3" unit="s" label="each picture shown" />
            <Stat value="5" unit="" label="runs each" />
          </div>
          <div className="cards3">
            <div className="pcard" style={{ borderColor: PL }}><div className="pcard-t" style={{ color: PL }}>Pleasant</div><p>Sports, nature, happy people, romance. Rated high on "how positive."</p></div>
            <div className="pcard" style={{ borderColor: NT }}><div className="pcard-t" style={{ color: NT }}>Neutral</div><p>Landscapes, everyday objects, people doing ordinary things.</p></div>
            <div className="pcard" style={{ borderColor: UP }}><div className="pcard-t" style={{ color: UP }}>Unpleasant</div><p>Threat, injury, attack scenes. Rated low on "how positive."</p></div>
          </div>
          <p className="fine">Pictures come from the IAPS, a research library where every image is pre-rated by hundreds of people — so "pleasant" isn't just one person's opinion.</p>
        </Section>

        {/* 3 — ROADMAP */}
        <Section id="map" n="3" kicker="The roadmap" title="The whole journey, before we zoom in">
          <p>
            Getting from "a photo on a screen" to "the brain knows how you feel" takes several steps.
            Here's the entire pipeline in nine plain stages. Each of the sections that follow walks
            through one of them.
          </p>
          <PipelineMap />
        </Section>

        {/* 4 — SIGNAL */}
        <Section id="signal" n="4" kicker="Steps 2–3 · the scanner" title="The brain's signal is slow, blurry, and messy">
          <p>
            An MRI scanner doesn't measure thoughts. It measures <strong>blood flow</strong>: when a
            patch of brain works harder, fresh blood arrives — but <em>late</em>, peaking about 6 seconds
            after the picture, then fading. On top of that, people breathe, fidget and drift, so the raw
            signal is noisy. Step one of the analysis is simply to <strong>clean it up</strong> (remove
            movement and slow drift). Step two is to account for that sluggish delay:
          </p>
          <Figure caption="The delayed blood-flow response to a single picture. Because the delay is so predictable, researchers can mathematically 'rewind' it to recover the true response to each picture.">
            <HrfChart />
          </Figure>
          <Callout tone="muted"><strong>Why it matters:</strong> knowing the exact delay lets the analysis untangle overlapping responses — the essential trick behind the next step.</Callout>
        </Section>

        {/* 5 — BETA */}
        <Section id="beta" n="5" kicker="Step 4 · the key move" title="Turn a wiggly signal into one number per picture">
          <p>
            Here's the move that makes everything else possible. Every picture produces a smear of blood
            flow that overlaps with its neighbours. Using the known delay shape, the analysis fits a model
            that hands back a <strong>single number for each picture</strong> — how strongly this patch of
            brain responded to <em>that specific photo</em>. 300 pictures → 300 numbers, per brain pixel.
          </p>
          <Figure caption="From one long overlapping signal to one clean 'response strength' per picture. This number is the raw material for everything that follows.">
            <BetaSwatches />
          </Figure>
          <Img src="/media/glm_convolution.png" real
               alt="Research figure: trial onsets convolved with the HRF to build the GLM predictor."
               caption="The actual method researchers use. The measured signal (top) is explained by lining up each picture's onset with the delay shape (bottom) — a 'general linear model'. Solving it yields the per-picture numbers." />
          <Img src="/media/design_matrix.png" real
               alt="Research figure: single-trial beta estimation, target trial kept separate from all others."
               caption="To get a clean number for one picture, that picture is modelled on its own while all the other pictures are lumped into a single 'everything else' term. Repeat for all 300 pictures." />
        </Section>

        {/* 6 — PATTERN TABLE */}
        <Section id="table" n="6" kicker="Step 5 · organise it" title="Stack the numbers into a pattern table">
          <p>
            Now line those numbers up into a table: <strong>one row per picture, one column per brain
            pixel</strong>. Read across a row and you get that picture's <em>fingerprint</em> — the exact
            pattern of activity it produced across the brain. The whole question becomes: do pleasant
            pictures leave a different fingerprint than neutral ones?
          </p>
          <Figure caption="Each row is a picture; each column is a tiny patch of visual cortex. Notice the pleasant and unpleasant rows have visibly different colour patterns from the neutral rows — that difference is exactly what we'll try to detect.">
            <PatternTable />
          </Figure>
          <Callout>
            <strong>The shift in thinking:</strong> older studies averaged a whole region into one
            "louder or quieter?" number and found little. Looking at the full <em>pattern</em> across
            pixels — a fingerprint, not a volume knob — is what reveals the emotion. This is called
            <strong> multi-voxel pattern analysis (MVPA)</strong>.
          </Callout>
        </Section>

        {/* 7 — REGION */}
        <Section id="region" n="7" kicker="Step 6 · pick a place to look" title="Zoom into one visual region at a time">
          <p>
            The brain is huge, so the analysis is done one region at a time. The stars of the show are the
            <strong> retinotopic visual areas</strong> at the back of the head — the first stops for signal
            coming from the eyes. The very first is <strong>V1</strong>. If emotion can be read even there,
            it means the feeling is present at the earliest possible moment of seeing.
          </p>
          <Figure caption="Signal flows from the eye to the visual cortex at the back of the brain. Researchers test 17 of these early visual regions, from V1 outward.">
            <BrainRegion />
          </Figure>
        </Section>

        {/* 8 — DECODE */}
        <Section id="decode" n="8" kicker="Step 7 · the computer" title="Teach a computer to tell the fingerprints apart">
          <p>
            Feed the fingerprints to a simple machine-learning model (a <em>support-vector machine</em>).
            Its whole job: look at a picture's pattern and guess <strong>pleasant or neutral?</strong> The
            way it learns is intuitive — plot each picture as a dot and find the best line that separates
            the two groups.
          </p>
          <Figure caption="A toy version with two brain pixels. Pleasant pictures (green) and neutral ones (blue) land in different areas; the computer draws the best dividing line. New pictures are guessed by which side they fall on.">
            <DecisionBoundary />
          </Figure>
          <p>
            If emotion leaves no trace, the guesses are right about <strong>50% of the time</strong> —
            pure coin-flip. Anything reliably above ~54% means real emotional information is in there.
          </p>
        </Section>

        {/* 9 — TRUST */}
        <Section id="trust" n="9" kicker="Step 8 · don't fool yourself" title="Make sure the score isn't a fluke">
          <p>
            A computer can "cheat" by memorising the pictures it was trained on. Two safeguards stop that.
            First, <strong>cross-validation</strong>: always test the computer on pictures it has never
            seen. The data is split into blocks; the model trains on most and is tested on the held-out
            one, rotating through every block.
          </p>
          <Figure caption="Five rounds. Each round, four blocks teach the computer (blue) and one fresh block tests it (orange). The score is the average over all rounds — so it only counts performance on unseen pictures.">
            <CrossVal />
          </Figure>
          <p>
            Second, the <strong>shuffle test</strong> (a permutation test): scramble the emotion labels so
            they're meaningless, and re-run everything thousands of times. That builds a picture of what
            "pure luck" looks like. The real score has to stick out well beyond that luck cloud to count.
          </p>
          <Figure caption="The pale bell is thousands of scores from shuffled, meaningless labels — luck clusters at 50%. The real score (red) sits far to the right. The chance of luck alone producing it is under 0.1%.">
            <PermutationHist />
          </Figure>
        </Section>

        {/* 10 — RESULTS */}
        <Section id="results" n="10" kicker="Step 9 · the answer" title="Emotion is readable across the visual brain">
          <p>
            The verdict: the computer could tell emotional scenes from neutral ones using <em>only</em>
            visual cortex — <strong>every region, including V1</strong>. The "neutral camera" isn't
            neutral; the feeling is already in the picture the eyes send up.
          </p>
          <Figure caption="Decoding accuracy for 17 visual regions (approximate group values). Every bar clears the 54% significance line — V1 highlighted. Hover a bar for its value.">
            <DecodingChart />
          </Figure>
          <Img src="/media/decoding_real.png" real
               alt="Group MVPA decoding accuracy bar chart, pleasant and unpleasant vs neutral, all regions above threshold."
               caption="This lab's own reproduction of the result. Every region beats chance (50%) and the significance line, for both pleasant-vs-neutral (top) and unpleasant-vs-neutral (bottom). Stars mark p < 0.001." />
          <p>
            Zooming into ventral visual cortex — a key target of feedback from emotion centres — the
            effect holds in <strong>every single person</strong> tested (real data from this lab's 20
            subjects):
          </p>
          <Figure caption="Real ventral-visual-cortex decoding accuracy, one dot per person. All 20 subjects sit above the 50% chance line for both comparisons.">
            <VvcDots />
          </Figure>
          <div className="stat-row">
            <Stat value={mean(VVC_PL).toFixed(1)} unit="%" label="avg pleasant-vs-neutral (20 people)" color={PL} />
            <Stat value={mean(VVC_UP).toFixed(1)} unit="%" label="avg unpleasant-vs-neutral (20 people)" color={UP} />
            <Stat value="20 / 20" unit="" label="people above chance" color="var(--accent)" />
          </div>
        </Section>

        {/* 11 — WHEN */}
        <Section id="when" n="11" kicker="The 2022 study · timing" title="When, to the millisecond, does the feeling show up?">
          <p>
            fMRI tells you <em>where</em> but is too slow for <em>when</em>. The EEG cap fills the gap,
            tracking the brain millisecond by millisecond. Decoding the EEG at each instant reveals a clear
            order after a picture flashes on:
          </p>
          <Figure caption="Timeline of affective scene processing. Basic seeing happens first (~90–100 ms); the emotional identity only becomes readable around 200–260 ms — then stays switched on for roughly two seconds.">
            <TimelineChart />
          </Figure>
          <div className="stat-row">
            <Stat value="~90" unit="ms" label="visual cortex registers the scene" color="var(--seq)" />
            <Stat value="~200" unit="ms" label="pleasant becomes decodable" color={PL} />
            <Stat value="~260" unit="ms" label="unpleasant becomes decodable" color={UP} />
            <Stat value="~2" unit="s" label="how long the emotion pattern persists" color="var(--accent)" />
          </div>
          <p>
            That two-second persistence is a big clue. A signal just passing through would flicker and
            vanish; one that lingers means brain areas are talking back and forth, holding it up. A
            "generalization map" tests this: train the decoder at one moment, test it at another. A broad
            bright square (not a thin diagonal) means the same pattern is reused over and over.
          </p>
          <Figure caption="Illustrative generalization map. Bright well off the diagonal = a pattern learned early still works much later: a stable, sustained representation, not a fleeting one.">
            <GeneralizationMap />
          </Figure>
        </Section>

        {/* 12 — RSA */}
        <Section id="rsa" n="12" kicker="The 2022 study · RSA" title="Matching two very different measurements">
          <p>
            EEG knows <em>when</em> but not <em>where</em>; fMRI knows <em>where</em> but not <em>when</em>.
            To combine them, researchers use <strong>representational similarity analysis (RSA)</strong> —
            and the intuition is simpler than the name.
          </p>
          <Callout>
            <strong>The idea:</strong> forget the raw signals. For each measurement, just ask "which
            pictures does the brain treat as <em>similar</em>, and which as <em>different</em>?" That gives
            each one a similarity map. If the EEG's map at, say, 90&nbsp;ms matches a region's fMRI map,
            that region was doing its work <em>at 90&nbsp;ms</em>. You borrow EEG's clock to time-stamp
            fMRI's map.
          </Callout>
          <div className="rsa-steps">
            <div className="rstep"><span>1</span><p>Build a "who-looks-like-whom" grid for every EEG time point.</p></div>
            <div className="rstep"><span>2</span><p>Build the same grid for each visual region's fMRI pattern.</p></div>
            <div className="rstep"><span>3</span><p>Slide the EEG grid across time; find when it best matches each region.</p></div>
            <div className="rstep"><span>4</span><p>That best-match moment = when the region did its perceptual work.</p></div>
          </div>
          <Img src="/media/rsa_real.png" real
               alt="RSA summary heatmap: onset times around 90-110 ms across early, ventral and dorsal visual cortex."
               caption="This lab's real RSA result. Right panel = onset time in milliseconds for three chunks of visual cortex (EVC = early, VVC = ventral/object, DVC = dorsal/spatial). Perceptual processing switches on around 90–110 ms — before the emotion-specific pattern forms at 200–260 ms. Seeing first, feeling a moment later." />
        </Section>

        {/* 13 — LPP */}
        <Section id="lpp" n="13" kicker="The 2021 study · the feedback wave" title="A brain wave that reveals the feedback loop">
          <p>
            EEG also carries a slow wave called the <strong>late positive potential (LPP)</strong>. Starting
            ~300&nbsp;ms after an emotional picture, it swells and lingers — much bigger for emotional scenes
            than neutral. Scientists read it as a signature of <em>reentry</em>: deeper emotion regions
            feeding signals back into the visual system.
          </p>
          <Figure caption="The LPP (illustrative waveform; relative sizes are the real values from Bo et al. 2021: neutral 0.78, pleasant 1.97, unpleasant 2.26 µV). Emotional pictures drive a bigger, longer wave.">
            <LppWaveform />
          </Figure>
          <p>
            The payoff: if that feedback is what plants emotion in visual cortex, people with a
            <strong> bigger LPP</strong> should have a <strong>more decodable</strong> visual pattern — and
            they do, specifically in <em>ventral</em> visual cortex, the target of the feedback:
          </p>
          <div className="scatter-row">
            <LppScatter R={0.69} color={UP} seed={7} label="Unpleasant vs neutral · R = 0.69, p = 0.0008" />
            <LppScatter R={0.50} color={PL} seed={13} label="Pleasant vs neutral · R = 0.50, p = 0.026" />
          </div>
          <p className="fine">Scatter clouds are illustrative (per-subject LPP isn't in this dataset); the R and p values are the paper's real reported statistics.</p>
          <p>
            A connectivity analysis pinned down the feedback's sources: the <strong>amygdala</strong> (the
            brain's threat detector) for unpleasant scenes, and frontal regions for pleasant ones. The
            harder the amygdala pushed into visual cortex, the better unpleasant scenes decoded there
            (R&nbsp;=&nbsp;0.66).
          </p>
          <Figure caption="The loop in one picture: a scene flows forward into visual cortex while emotion centres send signals back — sharpening the emotional pattern in the regions they target.">
            <ReentryDiagram />
          </Figure>
        </Section>

        {/* 14 — TAKEAWAY */}
        <Section id="takeaway" n="14" kicker="Why it matters" title="The visual brain feels along with you">
          <p>
            Put the three studies together and one story emerges. Emotional meaning isn't bolted on
            downstream — it's woven into the earliest visual picture of a scene, appears within a few
            hundred milliseconds and is held for seconds, and is shaped by emotion centres reaching back
            into the visual system.
          </p>
          <div className="take-grid">
            <div className="take"><div className="take-n">Where</div><p>Every visual region, down to V1, carries decodable emotional information.</p></div>
            <div className="take"><div className="take-n">When</div><p>Seeing at ~90–100 ms; feeling at ~200–260 ms; sustained ≈ 2 s.</p></div>
            <div className="take"><div className="take-n">How</div><p>Feedback from the amygdala and frontal cortex — indexed by the LPP — sharpens it.</p></div>
          </div>
          <p className="fine">Beyond the science, this reframes perception: what you see and how you feel about it aren't separate steps. The brain builds them together, from the very first glance.</p>
        </Section>

        <footer className="foot">
          <p>
            Based on <strong>Bo, Yin, Liu, Hu, Meyyappan, Kim, Keil &amp; Ding (2021)</strong>, "Decoding
            Neural Representations of Affective Scenes in Retinotopic Visual Cortex," <em>Cerebral
            Cortex</em>; and <strong>Bo, Cui, Yin, Hu, Hong, Kim, Keil &amp; Ding (2022)</strong>, "Decoding
            the temporal dynamics of affective scene processing," <em>NeuroImage</em>.
          </p>
          <p className="fine">
            An educational visualisation. The decoding bar chart, ventral-cortex dots and VVC averages use
            this lab's real data; the RSA and GLM figures are real analysis outputs. ERP waveforms, the
            generalization map and the LPP scatter clouds are illustrative reconstructions drawn to match
            the reported statistics.
          </p>
        </footer>
      </main>
    </div>
  );
}
