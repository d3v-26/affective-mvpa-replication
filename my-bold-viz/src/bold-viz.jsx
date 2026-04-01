import { useState, useRef, useEffect, useMemo, useCallback } from "react";

// ─── CONSTANTS ───────────────────────────────────────────────────────────────

const HRF_VALUES = [0.00,0.04,0.28,0.58,0.32,0.08,-0.02,-0.03,-0.01];
const voxelColors = { v1: "#f97316", v2: "#06b6d4", v3: "#a855f7", v4: "#eab308" };
const voxelLabels = { v1: "Voxel 1", v2: "Voxel 2", v3: "Voxel 3", v4: "Voxel 4" };
const condColors  = { Pleasant: "#22c55e", Neutral: "#6366f1", Unpleasant: "#ef4444" };

// ─── TRIAL DEFINITIONS (20 Pl + 20 Nt + 20 Up = 60 trials, 1 run) ───────────
// Onsets interleaved Pl→Nt→Up every 3 TRs, starting TR=2, across 206 TRs.
// mulberry32/gaussianRng are function declarations (hoisted) — safe to call here.

const trials = (() => {
  const plOnsets = [2,11,20,29,38,47,56,65,74,83,92,101,110,119,128,137,146,155,164,173];
  const ntOnsets = [5,14,23,32,41,50,59,68,77,86,95,104,113,122,131,140,149,158,167,176];
  const upOnsets = [8,17,26,35,44,53,62,71,80,89,98,107,116,125,134,143,152,161,170,179];
  const out = [];
  for (let i = 0; i < 20; i++) {
    out.push({ name:`Pl${i+1}`, label:"Pleasant",   onset:plOnsets[i], color:"#22c55e", condition:"Pleasant"   });
    out.push({ name:`Nt${i+1}`, label:"Neutral",    onset:ntOnsets[i], color:"#6366f1", condition:"Neutral"    });
    out.push({ name:`Up${i+1}`, label:"Unpleasant", onset:upOnsets[i], color:"#ef4444", condition:"Unpleasant" });
  }
  return out;
})();

// ─── HRF-CONVOLVED REGRESSORS (one per trial, length 206) ────────────────────

const allRegressors = (() => {
  const out = {};
  trials.forEach(trial => {
    const reg = Array(206).fill(0);
    HRF_VALUES.forEach((h, i) => { if (trial.onset - 1 + i < 206) reg[trial.onset - 1 + i] += h; });
    out[trial.name] = reg;
  });
  return out;
})();

// backward-compat alias (kept for reference)
// eslint-disable-next-line no-unused-vars
const regressors = { Pl1: allRegressors.Pl1, Nt1: allRegressors.Nt1, Up1: allRegressors.Up1 };

// ─── BOLD DATA (206 TRs × 4 voxels) ─────────────────────────────────────────
// Synthetic signal: baseline 1000 + HRF-weighted condition patterns + tiny noise.

const boldData = (() => {
  const rng = mulberry32(99);
  const patterns = {
    Pleasant:   [21.0, -8.5, 25.0, -6.5],
    Neutral:    [ 8.0, 10.5,  7.5, 11.0],
    Unpleasant: [24.5,-10.0, 29.0, -9.5],
  };
  const vKeys = ["v1","v2","v3","v4"];
  return Array.from({ length: 206 }, (_, t) => {
    const row = { t: t + 1 };
    vKeys.forEach((v, vi) => {
      let sig = 1000;
      trials.forEach(trial => {
        const lag = t - (trial.onset - 1);
        if (lag >= 0 && lag < HRF_VALUES.length) sig += patterns[trial.condition][vi] * HRF_VALUES[lag];
      });
      sig += gaussianRng(rng) * 0.3;
      row[v] = +sig.toFixed(3);
    });
    return row;
  });
})();

// ─── MOTION (206 TRs × 6 params: tX, tY, tZ, rX, rY, rZ) ───────────────────

const motion = Array.from({ length: 206 }, (_, t) => {
  const d = t / 205;
  return [0.012+d*0.009, 0.003+d*0.005, -0.005+d*0.004, 0.001+d*0.001, 0.000+d*0.001, 0.002+d*0.001];
});

// ─── LINEAR ALGEBRA ──────────────────────────────────────────────────────────

function buildDesignMatrix() {
  // Returns [206 × 67]: 60 trial regressors + 6 motion + 1 constant
  const X = [];
  for (let t = 0; t < 206; t++) {
    const taskCols = trials.map(tr => allRegressors[tr.name][t]);
    X.push([...taskCols, ...motion[t], 1.0]);
  }
  return X;
}

function transpose(M) {
  const rows = M.length, cols = M[0].length;
  const T = Array.from({ length: cols }, () => Array(rows).fill(0));
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) T[j][i] = M[i][j];
  return T;
}

function matMul(A, B) {
  const rA = A.length, cA = A[0].length, cB = B[0].length;
  const C = Array.from({ length: rA }, () => Array(cB).fill(0));
  for (let i = 0; i < rA; i++) for (let j = 0; j < cB; j++) for (let k = 0; k < cA; k++) C[i][j] += A[i][k] * B[k][j];
  return C;
}

function matVecMul(M, v) { return M.map(row => row.reduce((s, val, j) => s + val * v[j], 0)); }

function svdSolve(X, y) {
  const Xt = transpose(X);
  const XtX = matMul(Xt, X);
  const Xty = matVecMul(Xt, y);
  for (let i = 0; i < XtX.length; i++) XtX[i][i] += 1e-8;
  return solveLU(XtX, Xty);
}

function solveLU(A, b) {
  const n = A.length, M = A.map(r => [...r]), p = b.slice();
  for (let col = 0; col < n; col++) {
    let maxVal = Math.abs(M[col][col]), maxRow = col;
    for (let row = col + 1; row < n; row++) if (Math.abs(M[row][col]) > maxVal) { maxVal = Math.abs(M[row][col]); maxRow = row; }
    if (maxVal < 1e-15) continue;
    if (maxRow !== col) { [M[col], M[maxRow]] = [M[maxRow], M[col]]; [p[col], p[maxRow]] = [p[maxRow], p[col]]; }
    for (let row = col + 1; row < n; row++) { const f = M[row][col] / M[col][col]; for (let j = col; j < n; j++) M[row][j] -= f * M[col][j]; p[row] -= f * p[col]; }
  }
  const x = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) { let s = p[i]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = Math.abs(M[i][i]) > 1e-15 ? s / M[i][i] : 0; }
  return x;
}

function solveOLS(X, y) { return svdSolve(X, y); }
function lerp(a, b, t) { return a + (b - a) * t; }

function useContainerWidth(ref) {
  const [width, setWidth] = useState(800);
  useEffect(() => {
    const el = ref.current; if (!el) return;
    const update = () => setWidth(Math.floor(el.getBoundingClientRect().width));
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return width;
}

// ─── SEEDED RNG ──────────────────────────────────────────────────────────────

function mulberry32(seed) {
  return function() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// Generate Gaussian with Box-Muller
function gaussianRng(rng) {
  const u1 = rng(), u2 = rng();
  return Math.sqrt(-2 * Math.log(u1 || 1e-10)) * Math.cos(2 * Math.PI * u2);
}

// ─── SIMULATE MULTI-RUN BETA DATA ───────────────────────────────────────────
// In the real pipeline, 5 runs × 20 trials/condition = 100 trials per condition.
// We simulate 20 trials/condition across 5 runs for our 4-voxel toy ROI.

const SIMULATED_DATA = (() => {
  const rng = mulberry32(42);
  // True condition patterns (mean beta at each voxel) - matching our observed GLM patterns
  const patterns = {
    Pleasant:    [21.0, -8.5, 25.0, -6.5],   // V1,V3 up, V2,V4 down
    Neutral:     [8.0,  10.5, 7.5,  11.0],    // all voxels moderate up
    Unpleasant:  [24.5, -10.0, 29.0, -9.5],   // V1,V3 strongly up, V2,V4 down
  };
  const noise = 5.0; // within-condition variability
  const conditions = ["Pleasant", "Neutral", "Unpleasant"];
  const nTrialsPerCond = 20;
  const nRuns = 5;
  const trialsPerCondPerRun = nTrialsPerCond / nRuns; // 4

  const allTrials = [];
  for (const cond of conditions) {
    for (let run = 0; run < nRuns; run++) {
      for (let t = 0; t < trialsPerCondPerRun; t++) {
        const betas = patterns[cond].map(mu => mu + gaussianRng(rng) * noise);
        allTrials.push({ condition: cond, run: run + 1, betas, trialIdx: run * trialsPerCondPerRun + t + 1 });
      }
    }
  }
  return { allTrials, patterns, conditions, nTrialsPerCond };
})();

// ─── SVM HELPERS ─────────────────────────────────────────────────────────────

function zScore(data) {
  // data: array of [nSamples][nFeatures]
  const n = data.length, m = data[0].length;
  const means = Array(m).fill(0), stds = Array(m).fill(0);
  for (let j = 0; j < m; j++) {
    for (let i = 0; i < n; i++) means[j] += data[i][j];
    means[j] /= n;
    for (let i = 0; i < n; i++) stds[j] += (data[i][j] - means[j]) ** 2;
    stds[j] = Math.sqrt(stds[j] / n) || 1;
  }
  return { normalized: data.map(row => row.map((v, j) => (v - means[j]) / stds[j])), means, stds };
}

// Simple linear SVM via gradient descent
function trainLinearSVM(X, y, C = 1.0, lr = 0.01, epochs = 300) {
  const n = X.length, m = X[0].length;
  const w = Array(m).fill(0);
  let b = 0;
  for (let ep = 0; ep < epochs; ep++) {
    for (let i = 0; i < n; i++) {
      const dot = X[i].reduce((s, v, j) => s + v * w[j], 0) + b;
      const margin = y[i] * dot;
      if (margin < 1) {
        for (let j = 0; j < m; j++) w[j] += lr * (C * y[i] * X[i][j] - w[j] / n);
        b += lr * C * y[i];
      } else {
        for (let j = 0; j < m; j++) w[j] -= lr * w[j] / n;
      }
    }
    lr *= 0.998;
  }
  return { w, b };
}

function predictSVM(X, model) {
  return X.map(row => {
    const dot = row.reduce((s, v, j) => s + v * model.w[j], 0) + model.b;
    return dot >= 0 ? 1 : -1;
  });
}

function runCrossValidation(dataA, dataB, nFolds = 5) {
  // dataA: class +1, dataB: class -1
  // Returns { foldAccuracies, meanAccuracy, predictions }
  const rng = mulberry32(123);
  const allData = [
    ...dataA.map(d => ({ x: d, y: 1 })),
    ...dataB.map(d => ({ x: d, y: -1 })),
  ];
  // Shuffle
  for (let i = allData.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [allData[i], allData[j]] = [allData[j], allData[i]];
  }

  const foldSize = Math.floor(allData.length / nFolds);
  const foldAccuracies = [];
  const allPredictions = Array(allData.length).fill(0);

  for (let fold = 0; fold < nFolds; fold++) {
    const testStart = fold * foldSize;
    const testEnd = fold === nFolds - 1 ? allData.length : testStart + foldSize;
    const trainSet = [...allData.slice(0, testStart), ...allData.slice(testEnd)];
    const testSet = allData.slice(testStart, testEnd);

    // Z-score on train
    const trainX = trainSet.map(d => d.x);
    const { normalized: trainNorm, means, stds } = zScore(trainX);
    const trainY = trainSet.map(d => d.y);

    // Normalize test with train stats
    const testNorm = testSet.map(d => d.x.map((v, j) => (v - means[j]) / stds[j]));
    const testY = testSet.map(d => d.y);

    const model = trainLinearSVM(trainNorm, trainY);
    const preds = predictSVM(testNorm, model);

    let correct = 0;
    for (let i = 0; i < preds.length; i++) {
      if (preds[i] === testY[i]) correct++;
      allPredictions[testStart + i] = preds[i];
    }
    foldAccuracies.push(correct / preds.length);
  }

  const meanAccuracy = foldAccuracies.reduce((a, b) => a + b, 0) / nFolds;
  return { foldAccuracies, meanAccuracy, allPredictions, allData };
}

// ─── CANVAS CHARTS ───────────────────────────────────────────────────────────

function BoldChart({ data, highlighted, width = 1500, height = 450 }) {
  const canvasRef = useRef(null);
  const keys = ["v1", "v2", "v3", "v4"];
  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = height * dpr; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, height);
    const pad = { top: 20, right: 20, bottom: 50, left: 65 };
    const plotW = width - pad.left - pad.right, plotH = height - pad.top - pad.bottom;
    let allVals = data.flatMap(d => keys.map(k => d[k]));
    const yMin = Math.min(...allVals) - 1, yMax = Math.max(...allVals) + 1;
    const nTR = data.length;
    const xScale = (t) => pad.left + ((t - 1) / (nTR - 1)) * plotW;
    const yScale = (v) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;
    ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) { const yv = yMin + (i / 4) * (yMax - yMin); ctx.beginPath(); ctx.moveTo(pad.left, yScale(yv)); ctx.lineTo(width - pad.right, yScale(yv)); ctx.stroke(); }
    // Draw thin onset markers for all 60 trials (too dense to label individually)
    trials.forEach(trial => {
      const x = xScale(trial.onset);
      ctx.strokeStyle = trial.color + "55"; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke(); ctx.setLineDash([]);
    });
    keys.forEach((key) => {
      const isActive = highlighted === null || highlighted === key;
      ctx.strokeStyle = isActive ? voxelColors[key] : voxelColors[key] + "25"; ctx.lineWidth = isActive ? 2 : 1; ctx.beginPath();
      data.forEach((d, i) => { const px = xScale(d.t), py = yScale(d[key]); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }); ctx.stroke();
    });
    ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top + plotH); ctx.lineTo(width - pad.right, pad.top + plotH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + plotH); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.5)"; ctx.font = "11px 'DM Mono', monospace"; ctx.textAlign = "center";
    for (let t = 1; t <= nTR; t++) { if (t === 1 || t % 20 === 0) ctx.fillText(`${t}`, xScale(t), pad.top + plotH + 18); }
    ctx.fillText("Scan (TR)", width / 2, pad.top + plotH + 40);
    ctx.textAlign = "right";
    for (let i = 0; i < 5; i++) { const yv = yMin + (i / 4) * (yMax - yMin); ctx.fillText(yv.toFixed(0), pad.left - 8, yScale(yv) + 4); }
    ctx.save(); ctx.translate(14, pad.top + plotH / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText("Signal Intensity", 0, 0); ctx.restore();
  }, [data, highlighted, width, height]);
  return <canvas ref={canvasRef} style={{ width, height, maxWidth: "100%" }} />;
}

function HrfChart({ width = 1500, height = 380 }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = height * dpr; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, height);
    const pad = { top: 20, right: 20, bottom: 45, left: 65 };
    const plotW = width - pad.left - pad.right, plotH = height - pad.top - pad.bottom;
    const times = HRF_VALUES.map((_, i) => i * 1.98); const yMin = -0.1, yMax = 0.65;
    const xScale = (t) => pad.left + (t / times[times.length - 1]) * plotW;
    const yScale = (v) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;
    ctx.strokeStyle = "rgba(255,255,255,0.15)"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(pad.left, yScale(0)); ctx.lineTo(width - pad.right, yScale(0)); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(xScale(times[0]), yScale(0));
    HRF_VALUES.forEach((v, i) => ctx.lineTo(xScale(times[i]), yScale(v)));
    ctx.lineTo(xScale(times[times.length - 1]), yScale(0)); ctx.closePath();
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    grad.addColorStop(0, "rgba(99,102,241,0.3)"); grad.addColorStop(1, "rgba(99,102,241,0.02)"); ctx.fillStyle = grad; ctx.fill();
    ctx.strokeStyle = "#818cf8"; ctx.lineWidth = 2.5; ctx.beginPath();
    HRF_VALUES.forEach((v, i) => { const px = xScale(times[i]), py = yScale(v); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }); ctx.stroke();
    HRF_VALUES.forEach((v, i) => {
      ctx.fillStyle = "#818cf8"; ctx.beginPath(); ctx.arc(xScale(times[i]), yScale(v), 4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#1e1b2e"; ctx.beginPath(); ctx.arc(xScale(times[i]), yScale(v), 2, 0, Math.PI * 2); ctx.fill();
    });
    ctx.fillStyle = "rgba(255,255,255,0.5)"; ctx.font = "10px 'DM Mono', monospace"; ctx.textAlign = "center";
    ctx.fillText("peak \u2248 6s", xScale(5.94), yScale(0.58) - 10); ctx.fillText("undershoot", xScale(13.86), yScale(-0.03) - 10);
    ctx.font = "11px 'DM Mono', monospace";
    times.forEach((t, i) => { if (i % 2 === 0 || i === times.length - 1) ctx.fillText(t.toFixed(1) + "s", xScale(t), pad.top + plotH + 18); });
    ctx.fillText("Time (seconds)", width / 2, pad.top + plotH + 38);
    ctx.textAlign = "right"; [-0.05, 0.0, 0.2, 0.4, 0.6].forEach(v => ctx.fillText(v.toFixed(2), pad.left - 8, yScale(v) + 4));
    ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top + plotH); ctx.lineTo(width - pad.right, pad.top + plotH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + plotH); ctx.stroke();
  }, [width, height]);
  return <canvas ref={canvasRef} style={{ width, height, maxWidth: "100%" }} />;
}

function DesignMatrix({ width = 1500, height = 550 }) {
  const canvasRef = useRef(null);
  // 60 task cols + 6 motion + 1 constant = 67
  const nCols = 67, nRows = 206;
  const matrix = useMemo(() => buildDesignMatrix(), []);
  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = height * dpr; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, height);
    const pad = { top: 50, right: 20, bottom: 30, left: 50 };
    const cellW = (width - pad.left - pad.right) / nCols;
    const cellH = (height - pad.top - pad.bottom) / nRows;

    // Pre-compute per-column min/max for normalization
    const colMin = Array(nCols).fill(Infinity), colMax = Array(nCols).fill(-Infinity);
    matrix.forEach(row => row.forEach((v, c) => { if (v < colMin[c]) colMin[c] = v; if (v > colMax[c]) colMax[c] = v; }));

    matrix.forEach((row, r) => {
      row.forEach((val, c) => {
        const x = pad.left + c * cellW, y2 = pad.top + r * cellH;
        const intensity = colMax[c] === colMin[c] ? 0.5 : (val - colMin[c]) / (colMax[c] - colMin[c]);
        let color;
        if (c < 20)      { color = `rgb(${Math.round(lerp(18,34,intensity))},${Math.round(lerp(12,197,intensity))},${Math.round(lerp(28,94,intensity))})`;  } // Pl green
        else if (c < 40) { color = `rgb(${Math.round(lerp(18,99,intensity))},${Math.round(lerp(12,102,intensity))},${Math.round(lerp(28,241,intensity))})`; } // Nt purple
        else if (c < 60) { color = `rgb(${Math.round(lerp(18,239,intensity))},${Math.round(lerp(12,68,intensity))},${Math.round(lerp(28,68,intensity))})`;  } // Up red
        else if (c < 66) { const v2 = Math.round(lerp(18, 80, intensity)); color = `rgb(${v2+15},${v2+10},${v2})`; }                                        // motion
        else             { color = `rgb(60,55,75)`; }                                                                                                        // constant
        ctx.fillStyle = color; ctx.fillRect(x + 0.5, y2 + 0.5, cellW - 0.5, cellH - 0.5);
      });
    });

    // Column group headers
    const groups = [
      { label: "Pleasant (Pl1–Pl20)",   start: 0,  end: 20, color: "#22c55e" },
      { label: "Neutral (Nt1–Nt20)",    start: 20, end: 40, color: "#6366f1" },
      { label: "Unpleasant (Up1–Up20)", start: 40, end: 60, color: "#ef4444" },
      { label: "Motion ×6",             start: 60, end: 66, color: "#888" },
      { label: "const",                 start: 66, end: 67, color: "#aaa" },
    ];
    ctx.font = "bold 9px 'DM Mono', monospace"; ctx.textAlign = "center";
    groups.forEach(g => {
      const cx = pad.left + (g.start + g.end) / 2 * cellW;
      ctx.fillStyle = g.color;
      ctx.fillText(g.label, cx, pad.top - 26);
      // bracket line
      ctx.strokeStyle = g.color + "60"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.left + g.start * cellW, pad.top - 16);
      ctx.lineTo(pad.left + g.end * cellW, pad.top - 16); ctx.stroke();
    });
    ctx.font = "8px 'DM Mono', monospace"; ctx.fillStyle = "rgba(255,255,255,0.2)";
    ctx.fillText("Task Regressors (HRF-convolved)", pad.left + 30 * cellW, pad.top - 40);

    // Row labels every 20 TRs
    ctx.textAlign = "right"; ctx.font = "9px 'DM Mono', monospace"; ctx.fillStyle = "rgba(255,255,255,0.4)";
    for (let r = 0; r < nRows; r++) {
      if (r === 0 || (r + 1) % 20 === 0)
        ctx.fillText(`t=${r + 1}`, pad.left - 4, pad.top + r * cellH + cellH / 2 + 3);
    }

    // Group separator lines
    ctx.strokeStyle = "rgba(255,255,255,0.2)"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    [20, 40, 60, 66].forEach(col => {
      const sepX = pad.left + col * cellW;
      ctx.beginPath(); ctx.moveTo(sepX, pad.top); ctx.lineTo(sepX, pad.top + nRows * cellH); ctx.stroke();
    });
    ctx.setLineDash([]);
    ctx.strokeStyle = "rgba(255,255,255,0.1)"; ctx.lineWidth = 1;
    ctx.strokeRect(pad.left, pad.top, nCols * cellW, nRows * cellH);
  }, [width, height, matrix]);
  return <canvas ref={canvasRef} style={{ width, height, maxWidth: "100%" }} />;
}

// ─── GLM SOLVER ──────────────────────────────────────────────────────────────

function GlmSolver({ width = 1500 }) {
  const [selectedVoxel, setSelectedVoxel] = useState("v1");
  const X = useMemo(() => buildDesignMatrix(), []);
  const y = useMemo(() => boldData.map(d => d[selectedVoxel]), [selectedVoxel]);
  const beta = useMemo(() => solveOLS(X, y), [X, y]);
  const yHat = useMemo(() => X.map(row => row.reduce((s, val, j) => s + val * beta[j], 0)), [X, beta]);
  const residuals = useMemo(() => y.map((yi, i) => yi - yHat[i]), [y, yHat]);

  // Aggregate task betas: mean of 20 Pl, 20 Nt, 20 Up + 6 motion + 1 constant = 10 display items
  const meanPl = beta ? beta.slice(0, 20).reduce((a,b)=>a+b,0)/20 : 0;
  const meanNt = beta ? beta.slice(20,40).reduce((a,b)=>a+b,0)/20 : 0;
  const meanUp = beta ? beta.slice(40,60).reduce((a,b)=>a+b,0)/20 : 0;
  const displayBetas = beta ? [meanPl, meanNt, meanUp, ...beta.slice(60,67)] : Array(10).fill(0);
  const betaNames = ["\u03B2\u0305_Pl", "\u03B2\u0305_Nt", "\u03B2\u0305_Up", "\u03B2_tX", "\u03B2_tY", "\u03B2_tZ", "\u03B2_rX", "\u03B2_rY", "\u03B2_rZ", "\u03B2\u2080"];
  const betaClr = ["#22c55e", "#6366f1", "#ef4444", "#777", "#777", "#777", "#666", "#666", "#666", "#999"];

  const canvasRef = useRef(null); const chartH = 240;
  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = chartH * dpr; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, chartH);
    const pad = { top: 25, right: 20, bottom: 50, left: 65 };
    const plotW = width - pad.left - pad.right, plotH = chartH - pad.top - pad.bottom;
    const nTR = y.length;
    const allVals = [...y, ...yHat];
    const yMin = Math.min(...allVals) - 1, yMax = Math.max(...allVals) + 1;
    const xScale = (t) => pad.left + ((t - 1) / (nTR - 1)) * plotW;
    const yScale = (v) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;
    ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) { const yv = yMin + (i / 4) * (yMax - yMin); ctx.beginPath(); ctx.moveTo(pad.left, yScale(yv)); ctx.lineTo(width - pad.right, yScale(yv)); ctx.stroke(); }
    // Thin onset markers for all trials
    trials.forEach(trial => { const x = xScale(trial.onset); ctx.strokeStyle = trial.color + "40"; ctx.lineWidth = 0.5; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke(); ctx.setLineDash([]); });
    y.forEach((yi, i) => { const px = xScale(i + 1); ctx.strokeStyle = "rgba(255,100,100,0.25)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(px, yScale(yi)); ctx.lineTo(px, yScale(yHat[i])); ctx.stroke(); });
    ctx.strokeStyle = "#f472b6"; ctx.lineWidth = 2; ctx.beginPath();
    yHat.forEach((v, i) => { const px = xScale(i + 1), py = yScale(v); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }); ctx.stroke();
    ctx.strokeStyle = voxelColors[selectedVoxel]; ctx.lineWidth = 1.5; ctx.beginPath();
    y.forEach((v, i) => { const px = xScale(i + 1), py = yScale(v); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }); ctx.stroke();
    ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top + plotH); ctx.lineTo(width - pad.right, pad.top + plotH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + plotH); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.5)"; ctx.font = "11px 'DM Mono', monospace"; ctx.textAlign = "center";
    for (let t = 1; t <= nTR; t++) { if (t === 1 || t % 20 === 0) ctx.fillText(`${t}`, xScale(t), pad.top + plotH + 18); }
    ctx.fillText("Scan (TR)", width / 2, pad.top + plotH + 40);
    ctx.textAlign = "right"; for (let i = 0; i < 5; i++) { const yv = yMin + (i / 4) * (yMax - yMin); ctx.fillText(yv.toFixed(0), pad.left - 8, yScale(yv) + 4); }
    ctx.textAlign = "left"; ctx.font = "bold 11px 'DM Mono', monospace";
    ctx.fillStyle = voxelColors[selectedVoxel]; ctx.fillText("\u25CF y (observed)", pad.left + 8, pad.top + 2);
    ctx.fillStyle = "#f472b6"; ctx.fillText("\u25CF \u0177 = X\u03B2\u0302 (fitted)", pad.left + 160, pad.top + 2);
    ctx.fillStyle = "rgba(255,100,100,0.6)"; ctx.fillText("\u2502 \u03B5 (residual)", pad.left + 340, pad.top + 2);
  }, [y, yHat, selectedVoxel, width]);

  const residCanvasRef = useRef(null); const residH = 100;
  useEffect(() => {
    const canvas = residCanvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = residH * dpr; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, residH);
    const pad = { top: 15, right: 20, bottom: 25, left: 65 };
    const plotW = width - pad.left - pad.right, plotH = residH - pad.top - pad.bottom;
    const nTRr = residuals.length;
    const rMax = Math.max(...residuals.map(Math.abs)) * 1.2 || 1;
    const xScale = (t) => pad.left + ((t - 1) / (nTRr - 1)) * plotW;
    const yScale = (v) => pad.top + plotH / 2 - (v / rMax) * (plotH / 2);
    ctx.strokeStyle = "rgba(255,255,255,0.15)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, yScale(0)); ctx.lineTo(width - pad.right, yScale(0)); ctx.stroke();
    residuals.forEach((r, i) => {
      const px = xScale(i + 1), barW = Math.max(plotW / (nTRr * 1.2), 0.8);
      ctx.fillStyle = r > 0 ? "rgba(239,68,68,0.35)" : "rgba(99,102,241,0.35)";
      ctx.fillRect(px - barW / 2, Math.min(yScale(0), yScale(r)), barW, Math.abs(yScale(0) - yScale(r)));
    });
    ctx.fillStyle = "rgba(255,255,255,0.3)"; ctx.font = "10px 'DM Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText("\u03B5 = y \u2212 X\u03B2\u0302", pad.left + 4, pad.top + 4);
  }, [residuals, width]);

  // GLM matrix equation canvas: X_task [206×60] × β [60×1] = ŷ [206×1]
  const glmMatRef = useRef(null);
  const glmMatH = 260;
  useEffect(() => {
    const canvas = glmMatRef.current; if (!canvas) return;
    if (!beta || beta.some(v => !isFinite(v))) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = glmMatH * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, glmMatH);

    const nR = 206, bg = 12;
    const top = 44;
    const bandW = 44, cW = 30;
    const bandH = (glmMatH - top - 24);
    const cH = bandH / nR;
    const condBands = [
      { label: "Pl ×20", rgb: [34,197,94],  start: 0,  end: 20 },
      { label: "Nt ×20", rgb: [99,102,241], start: 20, end: 40 },
      { label: "Up ×20", rgb: [239,68,68],  start: 40, end: 60 },
    ];
    const vRGBsel2 = selectedVoxel==="v1"?[249,115,22]:selectedVoxel==="v2"?[6,182,212]:selectedVoxel==="v3"?[168,85,247]:[234,179,8];

    // ─ X_task [206×60] shown as 3 condition-averaged bands
    let x0 = 55;
    ctx.fillStyle = "#c7d2fe"; ctx.font = "bold 10px 'DM Mono',monospace"; ctx.textAlign = "center";
    ctx.fillText("X_task (task cols)", x0 + 1.5*bandW, top - 22);
    ctx.fillStyle = "rgba(255,255,255,0.2)"; ctx.font = "8px 'DM Mono',monospace";
    ctx.fillText("[206 \u00D7 60]", x0 + 1.5*bandW, top - 11);
    condBands.forEach((band, c) => {
      const bx2 = x0 + c * bandW;
      ctx.fillStyle = `rgb(${band.rgb.join(",")})`; ctx.font = "bold 8px 'DM Mono',monospace"; ctx.textAlign = "center";
      ctx.fillText(band.label, bx2 + bandW/2, top - 1);
      for (let r = 0; r < nR; r++) {
        let meanVal = 0;
        for (let ci = band.start; ci < band.end; ci++) meanVal += (allRegressors[trials[ci].name][r] || 0);
        meanVal /= 20;
        const intensity = Math.max(0, Math.min(1, (meanVal + 0.6) / 1.2));
        ctx.fillStyle = `rgb(${Math.round(lerp(bg,band.rgb[0],intensity))},${Math.round(lerp(bg,band.rgb[1],intensity))},${Math.round(lerp(bg,band.rgb[2],intensity))})`;
        ctx.fillRect(bx2+0.5, top+r*cH+0.5, bandW-1, Math.max(cH-0.5, 0.5));
      }
    });
    ctx.strokeStyle = "rgba(255,255,255,0.1)"; ctx.lineWidth=1; ctx.strokeRect(x0, top, 3*bandW, nR*cH);
    ctx.fillStyle="rgba(255,255,255,0.3)"; ctx.font="9px 'DM Mono',monospace"; ctx.textAlign="right";
    for (let r=0;r<nR;r++) if(r===0||(r+1)%40===0) ctx.fillText(`t=${r+1}`, x0-4, top+r*cH+cH/2+3);

    // ─ × operator
    let ox = x0 + 3*bandW + 16;
    ctx.fillStyle="rgba(255,255,255,0.35)"; ctx.font="16px 'DM Mono',monospace"; ctx.textAlign="center";
    ctx.fillText("\u00D7", ox, top + nR*cH/2 + 5);

    // ─ β [60×1] shown as 3 condition color bands
    const bx = ox + 20;
    const bCH = nR * cH / 60;
    ctx.fillStyle="#f472b6"; ctx.font="bold 10px 'DM Mono',monospace"; ctx.textAlign="center";
    ctx.fillText("\u03B2\u0302", bx + cW/2, top - 22);
    ctx.fillStyle="rgba(255,255,255,0.2)"; ctx.font="8px 'DM Mono',monospace";
    ctx.fillText("[60\u00D71]", bx + cW/2, top - 11);
    const bMax = Math.max(...beta.slice(0,60).map(Math.abs)) || 1;
    condBands.forEach(band => {
      for (let i = band.start; i < band.end; i++) {
        const bv = beta[i], t2 = (bv + bMax) / (2*bMax);
        ctx.fillStyle = `rgb(${Math.round(lerp(bg,band.rgb[0],t2))},${Math.round(lerp(bg,band.rgb[1],t2))},${Math.round(lerp(bg,band.rgb[2],t2))})`;
        ctx.fillRect(bx+0.5, top+i*bCH+0.5, cW-1, Math.max(bCH-0.5,0.5));
      }
      ctx.fillStyle = `rgb(${band.rgb.join(",")})`; ctx.font="7px 'DM Mono',monospace"; ctx.textAlign="left";
      ctx.fillText(band.label, bx+cW+2, top+(band.start+band.end)/2*bCH+3);
    });
    ctx.strokeStyle="rgba(244,114,182,0.25)"; ctx.strokeRect(bx, top, cW, 60*bCH);

    // ─ = operator
    ox = bx + cW + 55;
    ctx.fillStyle="rgba(255,255,255,0.35)"; ctx.font="16px 'DM Mono',monospace"; ctx.textAlign="center";
    ctx.fillText("=", ox, top + nR*cH/2 + 5);

    // ─ ŷ [206×1]
    const yx = ox + 20;
    const yMin2 = Math.min(...yHat), yMax2 = Math.max(...yHat);
    ctx.fillStyle = `rgb(${vRGBsel2.join(",")})`; ctx.font="bold 10px 'DM Mono',monospace"; ctx.textAlign="center";
    ctx.fillText("\u0177", yx + cW/2, top - 22);
    ctx.fillStyle="rgba(255,255,255,0.2)"; ctx.font="8px 'DM Mono',monospace";
    ctx.fillText("[206\u00D71]", yx + cW/2, top - 11);
    yHat.forEach((yv, r) => {
      const t = (yv - yMin2) / Math.max(yMax2 - yMin2, 0.001);
      ctx.fillStyle = `rgb(${Math.round(lerp(bg,vRGBsel2[0],t))},${Math.round(lerp(bg,vRGBsel2[1],t))},${Math.round(lerp(bg,vRGBsel2[2],t))})`;
      ctx.fillRect(yx+0.5, top+r*cH+0.5, cW-1, Math.max(cH-0.5,0.5));
    });
    ctx.strokeStyle="rgba(255,255,255,0.1)"; ctx.strokeRect(yx, top, cW, nR*cH);

    ctx.fillStyle="rgba(255,255,255,0.2)"; ctx.font="9px 'DM Mono',monospace"; ctx.textAlign="left";
    ctx.fillText("X\u1D40\u03B2\u0302 = \u0177  — task columns shown as condition-averaged bands; motion + constant omitted", 55, glmMatH-6);
  }, [beta, yHat, selectedVoxel, width]);

  if (!beta || beta.some(v => !isFinite(v))) return <div style={{ color: "#ef4444", padding: 20 }}>Numerical issue.</div>;

  return (
    <div>
      <div style={{ padding: "14px 20px", marginBottom: 20, borderRadius: 8, background: "linear-gradient(135deg, rgba(244,114,182,0.08), rgba(129,140,248,0.08))", border: "1px solid rgba(244,114,182,0.2)", textAlign: "center" }}>
        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 6, letterSpacing: 1, textTransform: "uppercase" }}>General Linear Model</div>
        <div style={{ fontSize: 22, fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700 }}>
          <span style={{ color: voxelColors[selectedVoxel] }}>y</span><span style={{ color: "rgba(255,255,255,0.3)", margin: "0 8px" }}>=</span>
          <span style={{ color: "#c7d2fe" }}>X</span><span style={{ color: "#f472b6" }}>&beta;</span>
          <span style={{ color: "rgba(255,255,255,0.3)", margin: "0 8px" }}>+</span><span style={{ color: "rgba(239,68,68,0.6)" }}>&epsilon;</span>
        </div>
        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", marginTop: 6 }}>Solve: <code style={{ color: "#f472b6" }}>&beta;&#770; = (X&#7511;X + &lambda;I)&#8315;&sup1; X&#7511;y</code></div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
        <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)" }}>Solve for:</span>
        {["v1","v2","v3","v4"].map(v => (
          <button key={v} onClick={() => setSelectedVoxel(v)} style={{ padding: "5px 14px", fontSize: 12, fontFamily: "inherit", border: `1px solid ${selectedVoxel === v ? voxelColors[v] : "rgba(255,255,255,0.1)"}`, background: selectedVoxel === v ? voxelColors[v] + "20" : "transparent", color: voxelColors[v], borderRadius: 5, cursor: "pointer", fontWeight: selectedVoxel === v ? 600 : 400 }}>{voxelLabels[v]}</button>
        ))}
      </div>
      <canvas ref={canvasRef} style={{ width, height: chartH, maxWidth: "100%" }} />
      <canvas ref={residCanvasRef} style={{ width, height: residH, marginTop: 4, maxWidth: "100%" }} />
      <div style={{ marginTop: 16 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: "rgba(255,255,255,0.6)", marginBottom: 8, fontFamily: "'Space Grotesk', sans-serif" }}>Estimated &beta;&#770; for {voxelLabels[selectedVoxel]} &nbsp;<span style={{ fontWeight: 400, fontSize: 10, color: "rgba(255,255,255,0.3)" }}>(task values = mean over 20 trials each)</span></div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 6 }}>
          {displayBetas.map((b, i) => (
            <div key={i} style={{ padding: "8px 6px", borderRadius: 6, textAlign: "center", background: i < 3 ? betaClr[i] + "12" : "rgba(255,255,255,0.03)", border: `1px solid ${i < 3 ? betaClr[i] + "30" : "rgba(255,255,255,0.06)"}` }}>
              <div style={{ fontSize: 9, color: betaClr[i], marginBottom: 3, fontWeight: 500 }}>{betaNames[i]}</div>
              <div style={{ fontSize: 13, fontWeight: 600, color: i < 3 ? "#e2e0f0" : "rgba(255,255,255,0.45)" }}>{b.toFixed(2)}</div>
            </div>
          ))}
        </div>
      </div>
      <div style={{ marginTop: 16 }}>
        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", fontWeight: 500, marginBottom: 8 }}>
          Matrix form: X&#7511;&beta;&#770; = &#375; &nbsp;&nbsp;(task columns only, color = voxel selected above)
        </div>
        <canvas ref={glmMatRef} style={{ width, height: glmMatH, maxWidth: "100%" }} />
      </div>

      <div style={{ marginTop: 16, padding: 16, background: "rgba(244,114,182,0.06)", border: "1px solid rgba(244,114,182,0.15)", borderRadius: 8, fontSize: 12, lineHeight: 1.7, color: "rgba(255,255,255,0.55)" }}>
        <span style={{ color: "#f472b6", fontWeight: 500 }}>Limitation &rarr; </span>
        This standard GLM gives only <strong>3 mean betas per voxel</strong> (one per condition; each averaging over 20 trials). That is not enough for trial-by-trial MVPA. The <strong>next step</strong> shows the Mumford method, which estimates one beta <em>per trial</em>, giving 300 data points per voxel (60 trials/run × 5 runs).
      </div>
    </div>
  );
}

// ─── MUMFORD BETA SERIES (STEP 2b) ───────────────────────────────────────────

function MumfordMethod({ width = 1500 }) {
  const [selectedTrial, setSelectedTrial] = useState(0);
  const mainRef = useRef(null);
  const betaRef = useRef(null);

  // All 60 trial definitions derived from global trials array
  const trialDefs = useMemo(() => trials.map(tr => ({
    name: `Trial ${trials.indexOf(tr)+1} · ${tr.condition}`,
    shortName: tr.name,
    onset: tr.onset,
    color: tr.color,
    condition: tr.condition,
  })), []);

  const buildReg = (onset) => {
    const reg = Array(206).fill(0);
    HRF_VALUES.forEach((h, i) => { if (onset - 1 + i < 206) reg[onset - 1 + i] += h; });
    return reg;
  };
  const allRegs = useMemo(() => trialDefs.map(t => buildReg(t.onset)), [trialDefs]);

  const col0 = allRegs[selectedTrial];
  const col1 = useMemo(() => Array(206).fill(0).map((_, t) =>
    allRegs.reduce((s, r, i) => i === selectedTrial ? s : s + r[t], 0)
  ), [allRegs, selectedTrial]);

  const boldV1 = useMemo(() => boldData.map(d => d.v1), []);

  // Solve Mumford GLM for every trial → one beta per trial
  const trialBetas = useMemo(() => trialDefs.map((_, ti) => {
    const c0 = allRegs[ti];
    const c1 = Array(206).fill(0).map((_, t) =>
      allRegs.reduce((s, r, i) => i === ti ? s : s + r[t], 0)
    );
    const X = boldData.map((_, t) => [c0[t], c1[t], 1.0]);
    return svdSolve(X, boldV1);  // [β_trial, β_others, β_const]
  }), [allRegs, boldV1]);

  // ── Canvas 1: BOLD + the two Mumford regressors + fit ──────────────────────
  useEffect(() => {
    const canvas = mainRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const H = 420;
    canvas.width = width * dpr; canvas.height = H * dpr;
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, H);

    const pad = { top: 30, right: 20, bottom: 44, left: 68 };
    const plotW = width - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;
    const nTR = 206;

    const xS = (t) => pad.left + (t / (nTR - 1)) * plotW;

    // Mumford predicted fit = β0*col0 + β1*col1 + β2
    const [b0, b1, b2] = trialBetas[selectedTrial];
    const fittedY = boldData.map((_, t) => b0 * col0[t] + b1 * col1[t] + b2);

    const allY = [...boldV1, ...fittedY];
    const yMin = Math.min(...allY) - 0.5;
    const yMax = Math.max(...allY) + 0.5;
    const yS = (v) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    // HRF scale to fit on the plot nicely
    const hrfAll = [...col0, ...col1];
    const hrfMax = Math.max(...hrfAll.map(Math.abs)) || 1;
    const hrfScale = (plotH * 0.22) / hrfMax;
    const hrfBaseline = pad.top + plotH * 0.85;
    const hS = (v) => hrfBaseline - v * hrfScale;

    // Grid
    ctx.strokeStyle = "rgba(255,255,255,0.05)"; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const yv = yMin + (i / 4) * (yMax - yMin);
      ctx.beginPath(); ctx.moveTo(pad.left, yS(yv)); ctx.lineTo(pad.left + plotW, yS(yv)); ctx.stroke();
    }

    // Shade selected trial; draw thin markers for all others
    trialDefs.forEach((tr, ti) => {
      const x0 = xS(tr.onset - 1);
      if (ti === selectedTrial) {
        ctx.fillStyle = tr.color + "18"; ctx.fillRect(x0, pad.top, xS(Math.min(tr.onset + 2, nTR-1)) - x0, plotH);
        ctx.strokeStyle = tr.color + "40"; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(x0, pad.top); ctx.lineTo(x0, pad.top + plotH); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = tr.color + "cc"; ctx.font = "bold 9px 'DM Mono', monospace"; ctx.textAlign = "center";
        ctx.fillText(tr.shortName, x0, pad.top + 10);
      } else {
        ctx.strokeStyle = tr.color + "30"; ctx.lineWidth = 0.5; ctx.setLineDash([2,3]);
        ctx.beginPath(); ctx.moveTo(x0, pad.top); ctx.lineTo(x0, pad.top + plotH); ctx.stroke(); ctx.setLineDash([]);
      }
    });

    // col1 HRF (others) — dim
    ctx.strokeStyle = "rgba(148,163,184,0.5)"; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
    ctx.beginPath();
    col1.forEach((v, t) => { t === 0 ? ctx.moveTo(xS(t), hS(v)) : ctx.lineTo(xS(t), hS(v)); });
    ctx.stroke(); ctx.setLineDash([]);

    // col0 HRF (trial of interest) — bright
    const trColor = trialDefs[selectedTrial].color;
    ctx.strokeStyle = trColor + "cc"; ctx.lineWidth = 2;
    ctx.beginPath();
    col0.forEach((v, t) => { t === 0 ? ctx.moveTo(xS(t), hS(v)) : ctx.lineTo(xS(t), hS(v)); });
    ctx.stroke();

    // HRF zero baseline
    ctx.strokeStyle = "rgba(255,255,255,0.08)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, hrfBaseline); ctx.lineTo(pad.left + plotW, hrfBaseline); ctx.stroke();

    // Mumford fit (dashed)
    ctx.strokeStyle = trColor + "80"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 3]);
    ctx.beginPath();
    fittedY.forEach((v, t) => { t === 0 ? ctx.moveTo(xS(t), yS(v)) : ctx.lineTo(xS(t), yS(v)); });
    ctx.stroke(); ctx.setLineDash([]);

    // BOLD signal (no dots for 206 TRs — too dense)
    ctx.strokeStyle = "rgba(255,255,255,0.75)"; ctx.lineWidth = 2;
    ctx.beginPath();
    boldV1.forEach((v, t) => { t === 0 ? ctx.moveTo(xS(t), yS(v)) : ctx.lineTo(xS(t), yS(v)); });
    ctx.stroke();

    // Axes
    ctx.strokeStyle = "rgba(255,255,255,0.2)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + plotH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top + plotH); ctx.lineTo(pad.left + plotW, pad.top + plotH); ctx.stroke();

    // Y labels (BOLD)
    ctx.fillStyle = "rgba(255,255,255,0.4)"; ctx.font = "9px 'DM Mono', monospace"; ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const yv = yMin + (i / 4) * (yMax - yMin);
      ctx.fillText(yv.toFixed(0), pad.left - 6, yS(yv) + 3);
    }
    ctx.save(); ctx.translate(14, pad.top + plotH / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center"; ctx.fillText("BOLD (a.u.)", 0, 0); ctx.restore();

    // X labels
    ctx.textAlign = "center";
    for (let t = 0; t < nTR; t += 20) ctx.fillText(`${t + 1}`, xS(t), pad.top + plotH + 16);
    ctx.fillText("Scan (TR)", pad.left + plotW / 2, pad.top + plotH + 34);

    // Legend
    const legItems = [
      { color: "rgba(255,255,255,0.75)", dash: false, label: "Measured BOLD (y)" },
      { color: trColor + "cc",           dash: false, label: `col₀: ${trialDefs[selectedTrial].name} HRF` },
      { color: "rgba(148,163,184,0.5)",  dash: true,  label: "col₁: All other trials HRF" },
      { color: trColor + "80",           dash: true,  label: "Mumford fit (β₀·col₀ + β₁·col₁ + β₂)" },
    ];
    legItems.forEach(({ color, dash, label }, i) => {
      const lx = pad.left + 8 + i * 310;
      const ly = pad.top + 20;
      ctx.strokeStyle = color; ctx.lineWidth = 1.5;
      if (dash) ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 22, ly); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255,255,255,0.45)"; ctx.font = "9px 'DM Mono', monospace"; ctx.textAlign = "left";
      ctx.fillText(label, lx + 26, ly + 3);
    });

    // Beta annotation
    ctx.fillStyle = trColor;
    ctx.font = "bold 11px 'DM Mono', monospace"; ctx.textAlign = "right";
    ctx.fillText(`β₀ = ${b0.toFixed(2)}  ← keep this`, pad.left + plotW - 4, pad.top + 22);
    ctx.fillStyle = "rgba(255,255,255,0.3)"; ctx.font = "10px 'DM Mono', monospace";
    ctx.fillText(`β₁ = ${b1.toFixed(2)}  (discard)`, pad.left + plotW - 4, pad.top + 38);

  }, [selectedTrial, width, col0, col1, boldV1, trialBetas, trialDefs]);

  // ── Canvas 2: beta per trial (bar chart) ───────────────────────────────────
  useEffect(() => {
    const canvas = betaRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const H = 160;
    canvas.width = width * dpr; canvas.height = H * dpr;
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, H);

    const pad = { top: 28, right: 20, bottom: 36, left: 68 };
    const plotW = width - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;
    // 60 bars, grouped by condition (Pl=green, Nt=purple, Up=red)
    const n = trialDefs.length; // 60
    const barW = Math.max((plotW / n) * 0.8, 1);
    const groupW = plotW / n;

    const allBetas = trialBetas.map(b => b[0]);
    const bMax = Math.max(...allBetas.map(Math.abs)) * 1.4 || 1;
    const yZ = pad.top + plotH / 2;
    const yS = (v) => yZ - (v / bMax) * (plotH / 2);

    // Zero line
    ctx.strokeStyle = "rgba(255,255,255,0.15)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, yZ); ctx.lineTo(pad.left + plotW, yZ); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.25)"; ctx.font = "9px 'DM Mono', monospace"; ctx.textAlign = "right";
    ctx.fillText("0", pad.left - 6, yZ + 3);
    ctx.fillText(`+${bMax.toFixed(0)}`, pad.left - 6, pad.top + 4);
    ctx.fillText(`-${bMax.toFixed(0)}`, pad.left - 6, pad.top + plotH);

    trialBetas.forEach(([betaVal], ti) => {
      const cx = pad.left + (ti + 0.5) * groupW;
      const x = cx - barW / 2;
      const color = trialDefs[ti].color;
      const isSel = ti === selectedTrial;
      const barTop = betaVal >= 0 ? yS(betaVal) : yZ;
      const barH2 = Math.abs(yS(betaVal) - yZ);
      ctx.fillStyle = color + (isSel ? "dd" : "44");
      ctx.fillRect(x, barTop, barW, barH2);
      if (isSel) {
        ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.strokeRect(x, barTop, barW, barH2);
        ctx.fillStyle = color; ctx.font = "bold 9px 'DM Mono', monospace"; ctx.textAlign = "center";
        ctx.fillText(`β=${betaVal.toFixed(1)}`, cx, betaVal >= 0 ? barTop - 4 : barTop + barH2 + 11);
      }
    });

    // Condition group labels at bottom
    const condGroups = [{ label:"Pl1-Pl20", cx: pad.left + 10*groupW, color:"#22c55e" }, { label:"Nt1-Nt20", cx: pad.left + 30*groupW, color:"#6366f1" }, { label:"Up1-Up20", cx: pad.left + 50*groupW, color:"#ef4444" }];
    condGroups.forEach(g => { ctx.fillStyle = g.color; ctx.font = "9px 'DM Mono', monospace"; ctx.textAlign = "center"; ctx.fillText(g.label, g.cx, pad.top + plotH + 16); });

    ctx.fillStyle = "rgba(255,255,255,0.5)"; ctx.font = "10px 'DM Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText("\u03B2\u2080 per trial (Voxel 1) — one GLM per trial, 60 total; selected trial highlighted", pad.left, 18);
  }, [selectedTrial, width, trialBetas, trialDefs]);

  // Per-trial design matrix canvas [206 × 3]: [col0 | col1 | const]
  const mumMatRef = useRef(null);
  const mumMatH = 320;
  useEffect(() => {
    const canvas = mumMatRef.current; if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = mumMatH * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, mumMatH);

    const nR = 206, bg = 12;
    const top = 44;
    const cW = (width - 75 - 20) / 3;
    const cH = (mumMatH - top - 24) / nR;
    const selC = trialDefs[selectedTrial].color;
    const selCond = trialDefs[selectedTrial].condition;
    const selRGB = selCond==="Pleasant"?[34,197,94]:selCond==="Neutral"?[99,102,241]:[239,68,68];

    const allVals = [...col0, ...col1];
    const gRange = Math.max(Math.abs(Math.min(...allVals)), Math.abs(Math.max(...allVals))) || 1;

    const colDefs = [
      { label: "col₀  HRF_trial_t", sublabel: "(trial of interest)", data: col0, rgb: selRGB, color: selC },
      { label: "col₁  HRF_others",  sublabel: "(all 59 other trials)",data: col1, rgb: [148,163,184], color: "#94a3b8" },
      { label: "col₂  constant",    sublabel: "(baseline intercept)", data: Array(206).fill(1), rgb: [80,80,90], color: "#555" },
    ];

    colDefs.forEach((col, c) => {
      ctx.fillStyle = col.color; ctx.font = "bold 10px 'DM Mono',monospace"; ctx.textAlign = "center";
      ctx.fillText(col.label, 75 + c*cW + cW/2, top - 22);
      ctx.fillStyle = "rgba(255,255,255,0.2)"; ctx.font = "8px 'DM Mono',monospace";
      ctx.fillText(col.sublabel, 75 + c*cW + cW/2, top - 11);

      col.data.forEach((val, r) => {
        const t = (val + gRange) / (2*gRange);
        const [R,G,B] = col.rgb;
        ctx.fillStyle = `rgb(${Math.round(lerp(bg,R,t))},${Math.round(lerp(bg,G,t))},${Math.round(lerp(bg,B,t))})`;
        ctx.fillRect(75+c*cW+0.5, top+r*cH+0.5, cW-1, Math.max(cH-0.5, 0.5));
      });
    });

    ctx.fillStyle = "rgba(255,255,255,0.3)"; ctx.font = "9px 'DM Mono',monospace"; ctx.textAlign = "right";
    for (let r = 0; r < nR; r++) {
      if (r === 0 || (r+1) % 40 === 0)
        ctx.fillText(`t=${r+1}`, 69, top+r*cH+cH/2+3);
    }

    ctx.strokeStyle = "rgba(255,255,255,0.1)"; ctx.lineWidth=1;
    ctx.strokeRect(75, top, 3*cW, nR*cH);
    ctx.setLineDash([3,3]);
    for (let c=1;c<3;c++){ctx.beginPath();ctx.moveTo(75+c*cW,top);ctx.lineTo(75+c*cW,top+nR*cH);ctx.stroke();}
    ctx.setLineDash([]);

    // Highlight col₀ (the keeper)
    ctx.strokeStyle = selC + "80"; ctx.lineWidth = 2;
    ctx.strokeRect(75+0.5, top, cW-1, nR*cH);

    ctx.fillStyle = "rgba(255,255,255,0.2)"; ctx.font = "9px 'DM Mono',monospace"; ctx.textAlign = "left";
    ctx.fillText(`X_t [206 \u00D7 3]  for ${trialDefs[selectedTrial].shortName} (trial ${selectedTrial+1}) — only \u03B2\u0302[col\u2080] is kept`, 75, mumMatH-6);
  }, [selectedTrial, col0, col1, width, trialDefs]);

  const selColor = trialDefs[selectedTrial].color;

  return (
    <div>
      <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 17, fontWeight: 600, margin: "0 0 4px" }}>
        Mumford Beta Series Method
      </h2>
      <p style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", margin: "0 0 16px" }}>
        Step 2b &middot; Mumford et al. (2012) &middot; One GLM per trial → one beta image per trial
      </p>

      {/* Goal */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 10, marginBottom: 16 }}>
        <div style={{ padding: 14, borderRadius: 8, background: "rgba(99,102,241,0.07)", border: "1px solid rgba(99,102,241,0.2)" }}>
          <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: 1, color: "#818cf8", fontWeight: 600, marginBottom: 6 }}>Goal</div>
          <div style={{ fontSize: 11, lineHeight: 1.7, color: "rgba(255,255,255,0.55)" }}>
            MVPA needs one brain pattern <em>per trial</em>. The standard GLM gives 1 beta per <em>condition</em> (3 numbers/voxel).
            We need <strong style={{ color: "#e2e0f0" }}>300 betas/voxel</strong> — one for each picture presentation.
          </div>
        </div>
        <div style={{ padding: 14, borderRadius: 8, background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.18)" }}>
          <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: 1, color: "#f87171", fontWeight: 600, marginBottom: 6 }}>Problem with naive approach</div>
          <div style={{ fontSize: 11, lineHeight: 1.7, color: "rgba(255,255,255,0.55)" }}>
            Giving every trial its own regressor creates <strong style={{ color: "#e2e0f0" }}>60 overlapping columns</strong>.
            Adjacent HRFs overlap (HRF lasts ~16 s, trials spaced ~5 s apart) → X'X near-singular → unstable betas.
          </div>
        </div>
        <div style={{ padding: 14, borderRadius: 8, background: "rgba(34,197,94,0.06)", border: "1px solid rgba(34,197,94,0.18)" }}>
          <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: 1, color: "#4ade80", fontWeight: 600, marginBottom: 6 }}>Mumford solution</div>
          <div style={{ fontSize: 11, lineHeight: 1.7, color: "rgba(255,255,255,0.55)" }}>
            Run <strong style={{ color: "#e2e0f0" }}>one separate GLM per trial</strong> with only 2 task regressors:
            <br /><code style={{ color: "#4ade80" }}>col₀</code> = this trial's HRF &nbsp;|&nbsp; <code style={{ color: "#94a3b8" }}>col₁</code> = all others merged.
            Only 2 columns → no collinearity.
          </div>
        </div>
      </div>

      {/* Input / Output row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 10, marginBottom: 16 }}>
        <div style={{ padding: "10px 14px", borderRadius: 8, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", fontSize: 11, color: "rgba(255,255,255,0.45)", lineHeight: 1.8 }}>
          <span style={{ color: "rgba(255,255,255,0.6)", fontWeight: 600 }}>Input: </span>
          Preprocessed BOLD <code style={{ color: "#c7d2fe" }}>y</code> [201 TRs × n_voxels] per run &nbsp;&middot;&nbsp;
          60 trial onset times per run &nbsp;&middot;&nbsp; 6 motion regressors
        </div>
        <div style={{ padding: "10px 14px", borderRadius: 8, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", fontSize: 11, color: "rgba(255,255,255,0.45)", lineHeight: 1.8 }}>
          <span style={{ color: "rgba(255,255,255,0.6)", fontWeight: 600 }}>Output: </span>
          <strong style={{ color: "#e2e0f0" }}>300 beta images</strong> per subject, each [53×63×46 voxels].
          Every voxel in image #k holds β for that trial. These are the MVPA feature vectors.
        </div>
      </div>

      {/* Formula */}
      <div style={{ padding: "10px 16px", marginBottom: 16, borderRadius: 8, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", fontFamily: "'DM Mono', monospace", fontSize: 12, lineHeight: 2, color: "rgba(255,255,255,0.55)" }}>
        <span style={{ color: "rgba(255,255,255,0.7)", fontWeight: 600 }}>For each trial t (repeated 300 times):</span>
        <br />
        <span style={{ color: "#c7d2fe" }}>X_t</span> = [<span style={{ color: selColor }}>HRF_trial_t</span> &nbsp;|&nbsp; <span style={{ color: "#94a3b8" }}>sum(HRF_all_others)</span> &nbsp;|&nbsp; <span style={{ color: "rgba(255,255,255,0.3)" }}>mot₁ … mot₆</span>] &nbsp;&nbsp;
        <span style={{ color: "rgba(255,255,255,0.3)" }}>[206 TRs × 8 cols]</span>
        <br />
        <span style={{ color: "#f472b6" }}>&beta;̂_t</span> = (X_t'X_t)⁻¹ X_t' y &nbsp;&rarr;&nbsp;
        keep only <strong style={{ color: selColor }}>&beta;̂_t[0]</strong> &nbsp;&nbsp;
        <span style={{ color: "rgba(255,255,255,0.3)" }}>(discard &beta;̂_t[1…7])</span>
      </div>

      {/* Per-trial design matrix */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", fontWeight: 500, marginBottom: 8 }}>
          Per-trial design matrix X_t &nbsp;&middot;&nbsp; select trial below to see how col&#8320; changes
        </div>
        <canvas ref={mumMatRef} style={{ width, height: mumMatH, maxWidth: "100%" }} />
      </div>

      {/* Trial selector — 60 trials via dropdown + quick condition buttons */}
      <div style={{ display: "flex", gap: 10, marginBottom: 12, alignItems: "center", flexWrap: "wrap" }}>
        <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)" }}>Select trial:</span>
        <select value={selectedTrial} onChange={e => setSelectedTrial(+e.target.value)} style={{
          background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.15)", color: trialDefs[selectedTrial].color,
          borderRadius: 5, padding: "4px 8px", fontSize: 11, fontFamily: "inherit", cursor: "pointer", maxWidth: "100%",
        }}>
          {trialDefs.map((tr, i) => (
            <option key={i} value={i} style={{ color: "#e2e0f0", background: "#1a1730" }}>{tr.shortName} — onset TR {tr.onset}</option>
          ))}
        </select>
        <span style={{ fontSize: 10, color: "rgba(255,255,255,0.25)" }}>or jump to first of each condition:</span>
        {[{label:"Pl1", idx:0, color:"#22c55e"},{label:"Nt1", idx:1, color:"#6366f1"},{label:"Up1", idx:2, color:"#ef4444"}].map(s => (
          <button key={s.label} onClick={() => setSelectedTrial(s.idx)} style={{
            padding: "3px 10px", fontSize: 10, fontFamily: "inherit", cursor: "pointer", borderRadius: 4,
            border: `1px solid ${s.color}50`, background: selectedTrial === s.idx ? s.color+"20" : "transparent", color: s.color,
          }}>{s.label}</button>
        ))}
      </div>

      {/* Main canvas */}
      <canvas ref={mainRef} style={{ width, height: 420, maxWidth: "100%" }} />

      {/* Beta bar chart */}
      <div style={{ marginTop: 12, fontSize: 11, color: "rgba(255,255,255,0.35)", marginBottom: 4 }}>
        After running all 60 GLMs (one per trial) → one &beta;&#8320; per trial per voxel:
      </div>
      <canvas ref={betaRef} style={{ width, height: 160, maxWidth: "100%" }} />

      {/* Key insight */}
      <div style={{ marginTop: 14, padding: 14, background: "rgba(34,197,94,0.06)", border: "1px solid rgba(34,197,94,0.15)", borderRadius: 8, fontSize: 11, lineHeight: 1.8, color: "rgba(255,255,255,0.5)" }}>
        <span style={{ color: "#4ade80", fontWeight: 600 }}>Why col₁ (others) stays as ONE column &rarr; </span>
        With only 2 task columns, <code>X_t'X_t</code> is well-conditioned — no collinearity. The "others" column absorbs all
        run-wide shared variance, leaving <code style={{ color: selColor }}>β₀</code> as a clean estimate of this trial's response amplitude.
        <br /><br />
        <span style={{ color: "rgba(255,255,255,0.3)", fontSize: 10 }}>
          In the real pipeline: 300 trials × 1 GLM each = 300 GLMs per subject. Each produces a whole-brain beta image
          [53×63×46 voxels]. These 300 images are the direct input to ROI masking and the SVM classifier.
        </span>
      </div>

      {/* Pipeline bridge */}
      <div style={{ marginTop: 12, display: "flex", gap: 4, flexWrap: "wrap", fontSize: 10, alignItems: "center" }}>
        {[
          { label: "Run 300 GLMs", color: "#f472b6" },
          { label: "→ 300 beta images", color: "#eab308" },
          { label: "→ ROI mask each image", color: "#fbbf24" },
          { label: "→ stack into [300 × n_voxels]", color: "#06b6d4" },
          { label: "→ SVM input ✓", color: "#22c55e" },
        ].map(({ label, color }, i) => (
          <span key={i} style={{ padding: "3px 8px", borderRadius: 3, background: `${color}10`, border: `1px solid ${color}25`, color: `${color}cc` }}>{label}</span>
        ))}
      </div>
    </div>
  );
}

// ─── STEP 4a: BETA FEATURE MATRIX ────────────────────────────────────────────

function BetaMatrix({ width = 1500 }) {
  const { allTrials, conditions, nTrialsPerCond } = SIMULATED_DATA;

  // Build flat [60 × 4] matrix ordered: all Pleasant, all Neutral, all Unpleasant
  const { matrix, rowConds } = useMemo(() => {
    const matrix = [], rowConds = [];
    for (const cond of conditions) {
      allTrials.filter(t => t.condition === cond).forEach(t => {
        matrix.push(t.betas.slice(0, 4));
        rowConds.push(cond);
      });
    }
    return { matrix, rowConds };
  }, []);

  const nRows = matrix.length; // 60
  const nCols = 4;

  const canvasRef = useRef(null);
  const H = 480;

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = H * dpr;
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, H);

    const condStripW = 14;
    const labelW = 54;
    const colLabelH = 28;
    const rowLabelW = 30;
    const pad = { top: colLabelH + 10, left: labelW + condStripW + 6, right: 140, bottom: 30 };
    const gridW = width - pad.left - pad.right;
    const gridH = H - pad.top - pad.bottom;
    const cellW = gridW / nCols;
    const cellH = gridH / nRows;

    // Global min/max for colour scale
    const allVals = matrix.flat();
    const vMin = Math.min(...allVals), vMax = Math.max(...allVals);
    const norm = v => (v - vMin) / (vMax - vMin || 1);
    const cellColor = v => {
      const t = norm(v);
      // RdBu: blue (low) → white (mid) → red (high)
      if (t < 0.5) {
        const f = t * 2;
        return `rgb(${Math.round(lerp(30, 230, f))},${Math.round(lerp(100, 230, f))},${Math.round(lerp(220, 230, f))})`;
      } else {
        const f = (t - 0.5) * 2;
        return `rgb(${Math.round(lerp(230, 220, f))},${Math.round(lerp(230, 50, f))},${Math.round(lerp(230, 50, f))})`;
      }
    };

    // Draw cells
    matrix.forEach((row, ri) => {
      row.forEach((val, ci) => {
        ctx.fillStyle = cellColor(val);
        ctx.fillRect(pad.left + ci * cellW, pad.top + ri * cellH, cellW - 0.5, cellH - 0.5);
      });
    });

    // Condition colour strip (left)
    rowConds.forEach((cond, ri) => {
      ctx.fillStyle = condColors[cond] + "cc";
      ctx.fillRect(pad.left - condStripW - 4, pad.top + ri * cellH, condStripW, cellH - 0.5);
    });

    // Condition group brackets + labels
    const groups = [
      { cond: "Pleasant",   start: 0,              count: nTrialsPerCond, color: "#22c55e" },
      { cond: "Neutral",    start: nTrialsPerCond,  count: nTrialsPerCond, color: "#6366f1" },
      { cond: "Unpleasant", start: nTrialsPerCond*2,count: nTrialsPerCond, color: "#ef4444" },
    ];
    groups.forEach(({ cond, start, count, color }) => {
      const y0 = pad.top + start * cellH;
      const y1 = pad.top + (start + count) * cellH;
      const mid = (y0 + y1) / 2;
      ctx.strokeStyle = color + "80"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(pad.left - condStripW - 10, y0); ctx.lineTo(pad.left - condStripW - 14, y0);
      ctx.lineTo(pad.left - condStripW - 14, y1); ctx.lineTo(pad.left - condStripW - 10, y1); ctx.stroke();
      ctx.fillStyle = color;
      ctx.font = "bold 10px 'DM Mono', monospace"; ctx.textAlign = "right";
      ctx.save(); ctx.translate(pad.left - condStripW - 18, mid); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center"; ctx.fillText(cond, 0, 0); ctx.restore();
      // trial index labels
      ctx.fillStyle = "rgba(255,255,255,0.25)"; ctx.font = "8px 'DM Mono', monospace"; ctx.textAlign = "right";
      ctx.fillText(`t${start + 1}`, pad.left - condStripW - 2, y0 + 8);
      ctx.fillText(`t${start + count}`, pad.left - condStripW - 2, y1 - 2);
    });

    // Column headers
    ["V1", "V2", "V3", "V4"].forEach((lbl, ci) => {
      ctx.fillStyle = voxelColors[`v${ci + 1}`];
      ctx.font = "bold 11px 'DM Mono', monospace"; ctx.textAlign = "center";
      ctx.fillText(lbl, pad.left + ci * cellW + cellW / 2, pad.top - 6);
    });
    ctx.fillStyle = "rgba(255,255,255,0.35)"; ctx.font = "9px 'DM Mono', monospace"; ctx.textAlign = "center";
    ctx.fillText("↑ voxels (features)", pad.left + gridW / 2, pad.top - 18);

    // Matrix dimension label
    ctx.fillStyle = "rgba(255,255,255,0.3)"; ctx.font = "10px 'DM Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText(`[${nRows} trials × ${nCols} voxels]`, pad.left, H - 8);

    // Grid lines
    ctx.strokeStyle = "rgba(0,0,0,0.25)"; ctx.lineWidth = 0.5;
    for (let ci = 1; ci < nCols; ci++) {
      ctx.beginPath();
      ctx.moveTo(pad.left + ci * cellW, pad.top);
      ctx.lineTo(pad.left + ci * cellW, pad.top + gridH);
      ctx.stroke();
    }
    // Condition dividers
    groups.slice(0, -1).forEach(({ start, count }) => {
      const y = pad.top + (start + count) * cellH;
      ctx.strokeStyle = "rgba(255,255,255,0.3)"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + gridW, y); ctx.stroke();
    });

    // Outer border
    ctx.strokeStyle = "rgba(255,255,255,0.15)"; ctx.lineWidth = 1;
    ctx.strokeRect(pad.left, pad.top, gridW, gridH);

    // Colorbar
    const cbX = pad.left + gridW + 20, cbY = pad.top, cbW = 16, cbH2 = gridH;
    for (let py = 0; py < cbH2; py++) {
      const t = 1 - py / cbH2;
      const v = vMin + t * (vMax - vMin);
      ctx.fillStyle = cellColor(v);
      ctx.fillRect(cbX, cbY + py, cbW, 1);
    }
    ctx.strokeStyle = "rgba(255,255,255,0.15)"; ctx.lineWidth = 1;
    ctx.strokeRect(cbX, cbY, cbW, cbH2);
    ctx.fillStyle = "rgba(255,255,255,0.45)"; ctx.font = "9px 'DM Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText(`${vMax.toFixed(0)}`, cbX + cbW + 4, cbY + 8);
    ctx.fillText(`0`, cbX + cbW + 4, cbY + cbH2 / 2);
    ctx.fillText(`${vMin.toFixed(0)}`, cbX + cbW + 4, cbY + cbH2 - 2);
    ctx.save(); ctx.translate(cbX + cbW + 42, cbY + cbH2 / 2); ctx.rotate(Math.PI / 2);
    ctx.textAlign = "center"; ctx.fillText("β value (a.u.)", 0, 0); ctx.restore();

    // Arrow "→ SVM"
    ctx.fillStyle = "#06b6d4"; ctx.font = "bold 10px 'DM Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText("Each row = one", cbX + cbW + 4, cbY + cbH2 + 12);
    ctx.fillText("data point for SVM", cbX + cbW + 4, cbY + cbH2 + 24);

  }, [width]);

  // Mean pattern table
  const means = conditions.map(cond => {
    const rows = matrix.filter((_, ri) => rowConds[ri] === cond);
    return [0,1,2,3].map(ci => rows.reduce((s, r) => s + r[ci], 0) / rows.length);
  });

  return (
    <div>
      <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 17, fontWeight: 600, margin: "0 0 4px" }}>
        Step 4a — Feature Matrix (Beta × Trial)
      </h2>
      <p style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", margin: "0 0 14px" }}>
        Stack all single-trial betas into one matrix: each row = one trial's voxel pattern
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 10, marginBottom: 14 }}>
        <div style={{ padding: "10px 14px", borderRadius: 8, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", fontSize: 11, color: "rgba(255,255,255,0.45)", lineHeight: 1.8 }}>
          <span style={{ color: "rgba(255,255,255,0.65)", fontWeight: 600 }}>Input: </span>
          300 single-trial beta images from Mumford step. Each image = [n_voxels] floats.
        </div>
        <div style={{ padding: "10px 14px", borderRadius: 8, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", fontSize: 11, color: "rgba(255,255,255,0.45)", lineHeight: 1.8 }}>
          <span style={{ color: "rgba(255,255,255,0.65)", fontWeight: 600 }}>Method: </span>
          Stack betas row-wise. Attach condition label to each row. Result: <code style={{ color: "#c7d2fe" }}>X</code> [300 × n_voxels], <code style={{ color: "#c7d2fe" }}>y</code> [300].
        </div>
        <div style={{ padding: "10px 14px", borderRadius: 8, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", fontSize: 11, color: "rgba(255,255,255,0.45)", lineHeight: 1.8 }}>
          <span style={{ color: "rgba(255,255,255,0.65)", fontWeight: 600 }}>Output: </span>
          Feature matrix <code style={{ color: "#c7d2fe" }}>X</code> [60 × 4] (toy) or [300 × ~400] (real). This is the classifier's input.
        </div>
      </div>

      <canvas ref={canvasRef} style={{ width, height: H, maxWidth: "100%" }} />

      {/* Mean pattern table */}
      <div style={{ marginTop: 16 }}>
        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 8, fontWeight: 500 }}>
          Mean β per condition — the "class centroid" each row cluster is pulled toward:
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "auto auto repeat(4, 1fr)", gap: "4px 10px", fontSize: 11, fontFamily: "'DM Mono', monospace", alignItems: "center" }}>
          <div />
          <div style={{ color: "rgba(255,255,255,0.3)", fontSize: 9 }}>n trials</div>
          {["V1","V2","V3","V4"].map((v,i) => (
            <div key={v} style={{ textAlign: "center", color: voxelColors[`v${i+1}`], fontWeight: 600 }}>{v}</div>
          ))}
          {conditions.map((cond, ci) => [
            <div key={`l${ci}`} style={{ color: condColors[cond], fontWeight: 700 }}>{cond}</div>,
            <div key={`n${ci}`} style={{ color: "rgba(255,255,255,0.3)", fontSize: 10, textAlign: "center" }}>20</div>,
            ...means[ci].map((m, vi) => (
              <div key={`${ci}-${vi}`} style={{ textAlign: "center", padding: "4px 2px", borderRadius: 4,
                background: m > 12 ? condColors[cond]+"22" : m < 0 ? "rgba(239,68,68,0.1)" : "rgba(255,255,255,0.03)",
                color: m > 0 ? "rgba(255,255,255,0.75)" : "#f87171",
              }}>{m.toFixed(1)}</div>
            )),
          ])}
        </div>
      </div>

      <div style={{ marginTop: 14, padding: 14, background: "rgba(6,182,212,0.06)", border: "1px solid rgba(6,182,212,0.18)", borderRadius: 8, fontSize: 11, lineHeight: 1.8, color: "rgba(255,255,255,0.5)" }}>
        <span style={{ color: "#22d3ee", fontWeight: 600 }}>What the matrix tells the SVM → </span>
        Each row is one <em>data point</em>. The SVM's job: find the hyperplane in this 4D voxel space
        that separates the green rows from the purple rows (PlNt) or red from purple (UpNt).
        Distinct row patterns per condition = decodable. Overlapping patterns = chance level.
      </div>
    </div>
  );
}

// ─── STEP 4b: ROI MASKING ────────────────────────────────────────────────────

function RoiMask({ width = 1500 }) {
  const { allTrials, conditions, nTrialsPerCond } = SIMULATED_DATA;
  const [step, setStep] = useState(0);  // 0=full brain, 1=mask overlay, 2=masked result
  const canvasRef = useRef(null);
  const H = 360;

  const nRows = 60;
  const nColsFull = 12;   // "full brain": 12 voxels shown (toy)
  const nColsMasked = 4;  // V1v ROI: 4 voxels
  const maskCols = [0, 1, 2, 3]; // which full-brain columns are in the ROI

  // Build [60 × 12] full-brain matrix (cols 0-3 = real betas, cols 4-11 = noise)
  const { fullMatrix, rowConds } = useMemo(() => {
    const fullMatrix = [], rowConds = [];
    const rng = mulberry32(9999);
    for (const cond of conditions) {
      allTrials.filter(t => t.condition === cond).forEach(t => {
        const noise = Array.from({ length: 8 }, () => gaussianRng(rng) * 4);
        fullMatrix.push([...t.betas.slice(0, 4), ...noise]);
        rowConds.push(cond);
      });
    }
    return { fullMatrix, rowConds };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = H * dpr;
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, H);

    const showMasked = step === 2;
    const showOverlay = step >= 1;
    const nCols = showMasked ? nColsMasked : nColsFull;

    const condStripW = 12;
    const padL = 80, padT = 36, padR = 20, padB = 30;
    const gridW = width - padL - padR;
    const gridH = H - padT - padB;
    const cellW = gridW / nCols;
    const cellH = gridH / nRows;

    const srcMatrix = showMasked
      ? fullMatrix.map(r => maskCols.map(c => r[c]))
      : fullMatrix;

    const allVals = srcMatrix.flat();
    const vMin = Math.min(...allVals), vMax = Math.max(...allVals);
    const cellColor = v => {
      const t = (v - vMin) / (vMax - vMin || 1);
      if (t < 0.5) {
        const f = t * 2;
        return `rgb(${Math.round(lerp(30,230,f))},${Math.round(lerp(100,230,f))},${Math.round(lerp(220,230,f))})`;
      }
      const f = (t - 0.5) * 2;
      return `rgb(${Math.round(lerp(230,220,f))},${Math.round(lerp(230,50,f))},${Math.round(lerp(230,50,f))})`;
    };

    // Draw cells
    srcMatrix.forEach((row, ri) => {
      row.forEach((val, ci) => {
        const isInMask = maskCols.includes(ci);
        ctx.globalAlpha = (showOverlay && !showMasked && !isInMask) ? 0.2 : 1.0;
        ctx.fillStyle = cellColor(val);
        ctx.fillRect(padL + ci * cellW, padT + ri * cellH, cellW - 0.5, cellH - 0.5);
      });
    });
    ctx.globalAlpha = 1;

    // Condition strip
    rowConds.forEach((cond, ri) => {
      ctx.fillStyle = condColors[cond] + "aa";
      ctx.fillRect(padL - condStripW - 3, padT + ri * cellH, condStripW, cellH - 0.5);
    });

    // Condition labels
    [
      { cond: "Pleasant", start: 0 }, { cond: "Neutral", start: nTrialsPerCond },
      { cond: "Unpleasant", start: nTrialsPerCond * 2 },
    ].forEach(({ cond, start }) => {
      const yMid = padT + (start + nTrialsPerCond / 2) * cellH;
      ctx.fillStyle = condColors[cond] + "cc";
      ctx.font = "9px 'DM Mono', monospace"; ctx.textAlign = "right";
      ctx.fillText(cond.slice(0, 6), padL - condStripW - 6, yMid + 3);
    });

    // Column headers
    const nColsToLabel = showMasked ? nColsMasked : nColsFull;
    for (let ci = 0; ci < nColsToLabel; ci++) {
      const isInMask = maskCols.includes(ci);
      ctx.fillStyle = isInMask ? (showOverlay ? "#fbbf24" : voxelColors[`v${ci+1}`]) : "rgba(255,255,255,0.2)";
      ctx.font = `${isInMask && showOverlay ? "bold " : ""}10px 'DM Mono', monospace`;
      ctx.textAlign = "center";
      ctx.fillText(isInMask ? `V${ci+1}` : `x${ci+1}`, padL + ci * cellW + cellW / 2, padT - 6);
    }

    // ROI mask overlay highlight
    if (showOverlay && !showMasked) {
      const x0 = padL + maskCols[0] * cellW;
      const x1 = padL + (maskCols[maskCols.length - 1] + 1) * cellW;
      ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 2.5; ctx.setLineDash([5, 3]);
      ctx.strokeRect(x0 - 1, padT - 1, x1 - x0 + 2, gridH + 2);
      ctx.setLineDash([]);
      ctx.fillStyle = "#fbbf24"; ctx.font = "bold 10px 'DM Mono', monospace"; ctx.textAlign = "center";
      ctx.fillText("V1v ROI", (x0 + x1) / 2, padT - 18);
      ctx.font = "9px 'DM Mono', monospace";
      ctx.fillText("← keep", (x0 + x1) / 2, padT + gridH + 14);
      // dim label for excluded
      ctx.fillStyle = "rgba(255,255,255,0.2)"; ctx.font = "9px 'DM Mono', monospace";
      ctx.fillText("← excluded (not in V1v mask)", padL + (nColsFull * 0.65) * cellW, padT + gridH + 14);
    }

    // Dim lines dividing conditions
    ctx.strokeStyle = "rgba(255,255,255,0.2)"; ctx.lineWidth = 1;
    [nTrialsPerCond, nTrialsPerCond * 2].forEach(r => {
      const y = padT + r * cellH;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + gridW, y); ctx.stroke();
    });

    // Outer border
    ctx.strokeStyle = "rgba(255,255,255,0.15)"; ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, gridW, gridH);

    // Dimension label
    ctx.fillStyle = "rgba(255,255,255,0.3)"; ctx.font = "10px 'DM Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText(showMasked ? `[${nRows} × ${nColsMasked}] — masked` : `[${nRows} × ${nColsFull}] — full brain (toy)`, padL, H - 8);

  }, [step, width]);

  const stepLabels = [
    { label: "① Full brain matrix", desc: "All voxels, all trials" },
    { label: "② Apply ROI mask", desc: "Highlight V1v columns" },
    { label: "③ Masked result", desc: "Keep only V1v voxels" },
  ];

  return (
    <div>
      <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 17, fontWeight: 600, margin: "0 0 4px" }}>
        Step 4b — ROI Masking
      </h2>
      <p style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", margin: "0 0 14px" }}>
        Apply the V1v binary mask to the full-brain feature matrix → keep only ROI voxels
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 10, marginBottom: 14 }}>
        <div style={{ padding: "10px 14px", borderRadius: 8, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", fontSize: 11, color: "rgba(255,255,255,0.45)", lineHeight: 1.8 }}>
          <span style={{ color: "rgba(255,255,255,0.65)", fontWeight: 600 }}>Input: </span>
          Feature matrix <code style={{ color: "#c7d2fe" }}>X</code> [300 × ~150k voxels] + binary ROI mask [~150k] with 1 for V1v voxels, 0 elsewhere.
        </div>
        <div style={{ padding: "10px 14px", borderRadius: 8, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", fontSize: 11, color: "rgba(255,255,255,0.45)", lineHeight: 1.8 }}>
          <span style={{ color: "rgba(255,255,255,0.65)", fontWeight: 600 }}>Method: </span>
          Select columns where mask == 1. Reduces dimensionality from ~150k to ~400 voxels for V1v. Repeated for each of 17 ROIs.
        </div>
        <div style={{ padding: "10px 14px", borderRadius: 8, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", fontSize: 11, color: "rgba(255,255,255,0.45)", lineHeight: 1.8 }}>
          <span style={{ color: "rgba(255,255,255,0.65)", fontWeight: 600 }}>Output: </span>
          17 masked matrices, each [300 × n_roi_voxels]. These feed directly into the SVM classifier, one ROI at a time.
        </div>
      </div>

      {/* Step buttons */}
      <div style={{ display: "flex", gap: 8, marginBottom: 12, alignItems: "center" }}>
        <span style={{ fontSize: 11, color: "rgba(255,255,255,0.35)" }}>Walk through:</span>
        {stepLabels.map(({ label, desc }, i) => (
          <button key={i} onClick={() => setStep(i)} style={{
            padding: "6px 14px", fontSize: 11, fontFamily: "inherit", cursor: "pointer", borderRadius: 5,
            border: `1px solid ${step === i ? "rgba(251,191,36,0.5)" : "rgba(255,255,255,0.08)"}`,
            background: step === i ? "rgba(251,191,36,0.1)" : "transparent",
            color: step === i ? "#fbbf24" : "rgba(255,255,255,0.35)",
          }}>
            {label}
            <div style={{ fontSize: 9, color: step === i ? "#fbbf24aa" : "rgba(255,255,255,0.2)", marginTop: 2 }}>{desc}</div>
          </button>
        ))}
      </div>

      <canvas ref={canvasRef} style={{ width, height: H, maxWidth: "100%" }} />

      <div style={{ marginTop: 14, padding: 14, background: "rgba(251,191,36,0.06)", border: "1px solid rgba(251,191,36,0.18)", borderRadius: 8, fontSize: 11, lineHeight: 1.8, color: "rgba(255,255,255,0.5)" }}>
        <span style={{ color: "#fbbf24", fontWeight: 600 }}>Why mask? → </span>
        The whole brain has ~150,000 voxels. Training an SVM on 150k features with only 300 samples would overfit severely
        (curse of dimensionality). The ROI mask reduces to ~400 voxels — the specific region whose patterns we want to test.
        This is done separately for each of the 17 retinotopic ROIs, producing 17 separate classification problems.
      </div>
    </div>
  );
}

// ─── CONDITION MATRICES & ROI MASK (STEP 3–4) ───────────────────────────────

function ConditionMatrices({ width = 1500 }) {
  const [showMask, setShowMask] = useState(false);
  const { allTrials, conditions, nTrialsPerCond } = SIMULATED_DATA;

  // Group by condition → matrix [nVoxels × nTrials]
  const condMatrices = useMemo(() => {
    const result = {};
    for (const cond of conditions) {
      const trials = allTrials.filter(t => t.condition === cond);
      // Each column is a trial's beta pattern across 4 voxels
      result[cond] = trials.map(t => t.betas);
    }
    return result;
  }, []);

  // Heatmap canvas
  const canvasRef = useRef(null);
  const heatH = 160;

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = heatH * dpr; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, heatH);

    const nCond = conditions.length;
    const gapBetween = 24;
    const totalGap = gapBetween * (nCond - 1);
    const matW = (width - 80 - totalGap) / nCond;
    const padLeft = 50;
    const padTop = 30;
    const nVox = showMask ? 4 : 8; // show "full brain" as 8 voxels, masked as 4
    const nTrials = nTrialsPerCond;
    const cellW = matW / nTrials;
    const cellH = (heatH - padTop - 20) / nVox;

    // Get global min/max for coloring
    let allVals = [];
    for (const cond of conditions) condMatrices[cond].forEach(t => t.forEach(v => allVals.push(v)));
    const vMin = Math.min(...allVals), vMax = Math.max(...allVals);

    conditions.forEach((cond, ci) => {
      const ox = padLeft + ci * (matW + gapBetween);

      // Condition label
      ctx.fillStyle = condColors[cond]; ctx.font = "bold 11px 'DM Mono', monospace"; ctx.textAlign = "center";
      ctx.fillText(cond, ox + matW / 2, padTop - 12);
      ctx.fillStyle = "rgba(255,255,255,0.25)"; ctx.font = "9px 'DM Mono', monospace";
      ctx.fillText(`[${nVox} \u00D7 ${nTrials}]`, ox + matW / 2, padTop - 2);

      // Draw cells
      const trialData = condMatrices[cond];
      for (let ti = 0; ti < nTrials; ti++) {
        for (let vi = 0; vi < nVox; vi++) {
          const val = vi < 4 ? trialData[ti][vi] : (trialData[ti][vi % 4] * 0.3 + gaussStatic(ci * 1000 + ti * 10 + vi));
          const intensity = (val - vMin) / (vMax - vMin || 1);
          const baseRgb = cond === "Pleasant" ? [34,197,94] : cond === "Neutral" ? [99,102,241] : [239,68,68];
          const r = Math.round(lerp(15, baseRgb[0], Math.max(0, Math.min(1, intensity))));
          const g = Math.round(lerp(10, baseRgb[1], Math.max(0, Math.min(1, intensity))));
          const b = Math.round(lerp(20, baseRgb[2], Math.max(0, Math.min(1, intensity))));
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.fillRect(ox + ti * cellW + 0.5, padTop + vi * cellH + 0.5, cellW - 1, cellH - 1);
        }
      }

      // Border
      ctx.strokeStyle = condColors[cond] + "40"; ctx.lineWidth = 1;
      ctx.strokeRect(ox, padTop, matW, nVox * cellH);

      // If mask applied, show ROI bracket
      if (showMask) {
        ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
        ctx.strokeRect(ox - 2, padTop - 2, matW + 4, nVox * cellH + 4); ctx.setLineDash([]);
      }
    });

    // Voxel labels
    ctx.fillStyle = "rgba(255,255,255,0.4)"; ctx.font = "9px 'DM Mono', monospace"; ctx.textAlign = "right";
    for (let vi = 0; vi < nVox; vi++) {
      const label = showMask ? `V${vi + 1}` : (vi < 4 ? `V1v.${vi + 1}` : `other.${vi - 3}`);
      ctx.fillText(label, padLeft - 6, padTop + vi * cellH + cellH / 2 + 3);
    }
  }, [showMask, width]);

  // Static noise helper (deterministic)
  function gaussStatic(seed) {
    const rng = mulberry32(seed);
    return gaussianRng(rng) * 3;
  }

  return (
    <div>
      <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 17, fontWeight: 600, margin: "0 0 4px" }}>Condition Matrices &amp; ROI Mask</h2>
      <p style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", margin: "0 0 16px" }}>
        Steps 3&ndash;4: Organize &beta;&#770; values by condition, then apply the V1v ROI mask
      </p>

      {/* Pipeline diagram */}
      <div style={{ padding: "12px 16px", marginBottom: 16, borderRadius: 8, background: "rgba(234,179,8,0.06)", border: "1px solid rgba(234,179,8,0.15)", fontSize: 11, lineHeight: 1.7, color: "rgba(255,255,255,0.5)" }}>
        <span style={{ color: "#eab308", fontWeight: 500 }}>Step 3:</span> Take each trial's &beta;&#770; from the GLM &rarr; group by condition &rarr; matrix [nVoxels &times; nTrials]
        <br />
        <span style={{ color: "#fbbf24", fontWeight: 500 }}>Step 4:</span> Apply ROI binary mask &rarr; keep only V1v voxels (4 out of ~150k)
      </div>

      {/* Toggle */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <button onClick={() => setShowMask(false)} style={{ padding: "6px 16px", fontSize: 11, fontFamily: "inherit", borderRadius: 5, cursor: "pointer", border: `1px solid ${!showMask ? "rgba(234,179,8,0.4)" : "rgba(255,255,255,0.08)"}`, background: !showMask ? "rgba(234,179,8,0.12)" : "transparent", color: !showMask ? "#fbbf24" : "rgba(255,255,255,0.35)" }}>
          Full Brain (8 voxels)
        </button>
        <button onClick={() => setShowMask(true)} style={{ padding: "6px 16px", fontSize: 11, fontFamily: "inherit", borderRadius: 5, cursor: "pointer", border: `1px solid ${showMask ? "rgba(234,179,8,0.4)" : "rgba(255,255,255,0.08)"}`, background: showMask ? "rgba(234,179,8,0.12)" : "transparent", color: showMask ? "#fbbf24" : "rgba(255,255,255,0.35)" }}>
          V1v ROI Masked (4 voxels)
        </button>
      </div>

      <canvas ref={canvasRef} style={{ width, height: heatH, maxWidth: "100%" }} />

      {/* Pattern summary table */}
      <div style={{ marginTop: 16 }}>
        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 8, fontWeight: 500 }}>Mean &beta;&#770; pattern per condition (V1v ROI)</div>
        <div style={{ display: "grid", gridTemplateColumns: "auto repeat(4, 1fr)", gap: "4px 8px", fontSize: 11, fontFamily: "'DM Mono', monospace" }}>
          <div style={{ color: "rgba(255,255,255,0.3)" }}></div>
          {[1,2,3,4].map(i => <div key={i} style={{ textAlign: "center", color: voxelColors[`v${i}`], fontWeight: 500 }}>V{i}</div>)}
          {conditions.map(cond => {
            const mean = [0,1,2,3].map(vi => {
              const vals = condMatrices[cond].map(t => t[vi]);
              return vals.reduce((a, b) => a + b, 0) / vals.length;
            });
            return [
              <div key={cond} style={{ color: condColors[cond], fontWeight: 600 }}>{cond.slice(0, 6)}</div>,
              ...mean.map((m, i) => (
                <div key={`${cond}-${i}`} style={{
                  textAlign: "center", padding: "3px 0", borderRadius: 3,
                  background: `${condColors[cond]}${m > 10 ? "20" : m < 0 ? "10" : "08"}`,
                  color: m > 0 ? "rgba(255,255,255,0.7)" : "rgba(239,68,68,0.7)",
                }}>{m.toFixed(1)}</div>
              ))
            ];
          })}
        </div>
      </div>

      <div style={{ marginTop: 16, padding: 16, background: "rgba(234,179,8,0.06)", border: "1px solid rgba(234,179,8,0.15)", borderRadius: 8, fontSize: 12, lineHeight: 1.7, color: "rgba(255,255,255,0.55)" }}>
        <span style={{ color: "#fbbf24", fontWeight: 500 }}>Key observation &rarr; </span>
        Pleasant and Unpleasant share a similar pattern (V1,V3 high / V2,V4 low) but with different magnitudes. Neutral is distinct (all voxels moderately positive). The SVM classifier in the next step will learn to separate these spatial patterns.
      </div>
    </div>
  );
}

// ─── SVM FEATURE MATRIX SUB-COMPONENT ───────────────────────────────────────

function SVMMatrix({ comparison, comparisons, width = 800 }) {
  const { allTrials } = SIMULATED_DATA;
  const [fold, setFold] = useState(0);
  const canvasRef = useRef(null);
  const H = 580;
  const nVoxels = 4;
  const nFolds = 5;

  const comp = comparisons[comparison];
  const allRows = useMemo(() => {
    const dataA = allTrials.filter(t => t.condition === comp.a).map(t => ({ ...t, cls: 1 }));
    const dataB = allTrials.filter(t => t.condition === comp.b).map(t => ({ ...t, cls: -1 }));
    return [...dataA, ...dataB];
  }, [comparison]);

  const nTrials = allRows.length;        // 40
  const foldSize = Math.floor(nTrials / nFolds);  // 8
  const testStart = fold * foldSize;
  const testEnd   = testStart + foldSize;

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = H * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, H);

    const vCols = [voxelColors.v1, voxelColors.v2, voxelColors.v3, voxelColors.v4];
    const allBetas = allRows.flatMap(r => r.betas);
    const bRange = Math.max(Math.abs(Math.min(...allBetas)), Math.abs(Math.max(...allBetas)));

    function rdBu(b, alpha) {
      const t = Math.max(-1, Math.min(1, b / bRange));
      let r_, g_, bl_;
      if (t >= 0) { r_ = Math.round(239*t + 139*(1-t)); g_ = Math.round(68*t + 92*(1-t)); bl_ = Math.round(68*t + 246*(1-t)); }
      else        { const s=-t; r_ = Math.round(59*s + 139*(1-s)); g_ = Math.round(130*s + 92*(1-s)); bl_ = Math.round(246*(1)); }
      return `rgba(${r_},${g_},${bl_},${alpha})`;
    }

    // ── LEFT PANEL: [nVoxels × nTrials] — stored format ───────────────────────
    // Rows = voxels (4), Cols = trials (40) — cell width scales with available space
    const rpW = Math.max(120, Math.round(width * 0.22));   // right panel fixed width
    const arrowW = 80;
    const lpW = width - 65 - arrowW - rpW - 10;           // remaining space for left panel
    const lpCellW = Math.max(6, Math.floor(lpW / nTrials));
    const rpCellW = Math.max(20, Math.floor((rpW - 30) / nVoxels));
    const LP = { x: 65, y: 95, cellW: lpCellW, cellH: 48 };
    // Panel title
    ctx.fillStyle = "rgba(255,255,255,0.6)"; ctx.font = "bold 11px 'DM Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText("BEFORE TRANSPOSE", LP.x, LP.y - 34);
    ctx.fillStyle = "rgba(255,255,255,0.25)"; ctx.font = "9px 'DM Mono', monospace";
    ctx.fillText("Pl#.mat from extract_betas.m — rows=voxels, cols=trials", LP.x, LP.y - 20);
    // Condition color bar along top of columns
    allRows.forEach((row, j) => {
      ctx.fillStyle = condColors[row.condition] + "80";
      ctx.fillRect(LP.x + j * LP.cellW, LP.y - 13, LP.cellW - 1, 10);
    });
    // Condition label under bar
    ctx.fillStyle = "rgba(255,255,255,0.2)"; ctx.font = "8px 'DM Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText("← trials (grouped by condition) →", LP.x, LP.y + nVoxels * LP.cellH + 16);
    // Cells
    for (let v = 0; v < nVoxels; v++) {
      ctx.fillStyle = vCols[v]; ctx.font = "bold 10px 'DM Mono', monospace"; ctx.textAlign = "right";
      ctx.fillText(["V1","V2","V3","V4"][v], LP.x - 7, LP.y + v * LP.cellH + LP.cellH / 2 + 4);
      for (let j = 0; j < nTrials; j++) {
        ctx.fillStyle = rdBu(allRows[j].betas[v], 0.85);
        ctx.fillRect(LP.x + j * LP.cellW, LP.y + v * LP.cellH, LP.cellW - 1, LP.cellH - 1);
      }
    }
    // Shape label
    ctx.fillStyle = "rgba(255,255,255,0.3)"; ctx.font = "9px 'DM Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText(`shape: [${nVoxels} \u00D7 ${nTrials}]`, LP.x, LP.y + nVoxels * LP.cellH + 30);

    // ── TRANSPOSE ARROW ──────────────────────────────────────────────────────
    const lpEndX  = LP.x + nTrials * LP.cellW;
    const lpMidY  = LP.y + nVoxels * LP.cellH / 2;
    const ax1 = lpEndX + 10, ax2 = ax1 + arrowW - 20;
    ctx.strokeStyle = "rgba(199,210,254,0.5)"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(ax1, lpMidY); ctx.lineTo(ax2, lpMidY); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ax2-9, lpMidY-5); ctx.lineTo(ax2, lpMidY); ctx.lineTo(ax2-9, lpMidY+5); ctx.stroke();
    ctx.fillStyle = "#c7d2fe"; ctx.font = "bold 13px 'DM Mono', monospace"; ctx.textAlign = "center";
    ctx.fillText("\u1D40", (ax1+ax2)/2, lpMidY - 10);
    ctx.fillStyle = "rgba(255,255,255,0.3)"; ctx.font = "9px 'DM Mono', monospace";
    ctx.fillText("transpose", (ax1+ax2)/2, lpMidY + 20);

    // ── RIGHT PANEL: [nTrials × nVoxels] — SVM format ────────────────────────
    // Rows = trials (40), Cols = voxels (4)
    const RP = { x: ax2 + 18, y: 95, cellW: rpCellW, cellH: 11 };
    // Panel title
    ctx.fillStyle = "rgba(255,255,255,0.6)"; ctx.font = "bold 11px 'DM Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText("AFTER TRANSPOSE  \u2192  SVM INPUT", RP.x, RP.y - 34);
    ctx.fillStyle = "rgba(255,255,255,0.25)"; ctx.font = "9px 'DM Mono', monospace";
    ctx.fillText("rows=trials (samples), cols=voxels (features)", RP.x, RP.y - 20);
    // Column headers (voxels)
    ["V1","V2","V3","V4"].forEach((lbl, j) => {
      ctx.fillStyle = vCols[j]; ctx.font = "bold 10px 'DM Mono', monospace"; ctx.textAlign = "center";
      ctx.fillText(lbl, RP.x + j * RP.cellW + RP.cellW / 2, RP.y - 7);
    });
    // Cells
    allRows.forEach((row, i) => {
      const isTest = i >= testStart && i < testEnd;
      const y = RP.y + i * RP.cellH;
      ctx.fillStyle = condColors[row.condition] + (isTest ? "70" : "35");
      ctx.fillRect(RP.x - 12, y, 10, RP.cellH - 0.5);
      row.betas.forEach((b, j) => {
        ctx.fillStyle = rdBu(b, isTest ? 0.28 : 0.82);
        ctx.fillRect(RP.x + j * RP.cellW, y, RP.cellW - 1, RP.cellH - 0.5);
      });
    });
    // Test fold highlight
    ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 1.5;
    ctx.strokeRect(RP.x - 0.5, RP.y + testStart * RP.cellH - 0.5, nVoxels * RP.cellW + 1, foldSize * RP.cellH + 1);
    // TRAIN/TEST labels
    const lx = RP.x - 26;
    if (testStart > 0) { ctx.save(); ctx.translate(lx, RP.y + (testStart/2)*RP.cellH); ctx.rotate(-Math.PI/2); ctx.fillStyle="rgba(255,255,255,0.25)"; ctx.font="8px 'DM Mono',monospace"; ctx.textAlign="center"; ctx.fillText("TRAIN",0,0); ctx.restore(); }
    ctx.save(); ctx.translate(lx, RP.y + (testStart + foldSize/2)*RP.cellH); ctx.rotate(-Math.PI/2); ctx.fillStyle="#fbbf24"; ctx.font="bold 8px 'DM Mono',monospace"; ctx.textAlign="center"; ctx.fillText(`TEST F${fold+1}`,0,0); ctx.restore();
    if (testEnd < nTrials) { ctx.save(); ctx.translate(lx, RP.y + (testEnd + (nTrials-testEnd)/2)*RP.cellH); ctx.rotate(-Math.PI/2); ctx.fillStyle="rgba(255,255,255,0.25)"; ctx.font="8px 'DM Mono',monospace"; ctx.textAlign="center"; ctx.fillText("TRAIN",0,0); ctx.restore(); }
    // Shape label
    ctx.fillStyle = "rgba(255,255,255,0.3)"; ctx.font = "9px 'DM Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText(`shape: [${nTrials} \u00D7 ${nVoxels}]  train: ${nTrials-foldSize}  test: ${foldSize}`, RP.x, RP.y + nTrials*RP.cellH + 16);
  }, [fold, comparison, allRows, width]);

  return (
    <div style={{ marginBottom: 20 }}>
      {/* Explanation cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 8, marginBottom: 14 }}>
        <div style={{ padding: "10px 14px", borderRadius: 7, background: "rgba(129,140,248,0.06)", border: "1px solid rgba(129,140,248,0.2)" }}>
          <div style={{ fontSize: 9, color: "#818cf8", fontWeight: 600, letterSpacing: 1.5, marginBottom: 5, textTransform: "uppercase" }}>Why transpose?</div>
          <div style={{ fontSize: 11, color: "rgba(255,255,255,0.55)", lineHeight: 1.7 }}>
            <code style={{ color: "#c7d2fe" }}>extract_betas.m</code> saves betas as <code style={{ color: "#c7d2fe" }}>[nVoxels &times; nTrials]</code> — each <em>column</em> is one trial, natural for neuroscience tools.
            The SVM (scikit-learn / LIBSVM) expects <code style={{ color: "#22c55e" }}>[nSamples &times; nFeatures]</code> — each <em>row</em> must be one sample.
            Transposing flips trials onto rows and voxels onto columns so every row = one data point the classifier can train on.
          </div>
        </div>
        <div style={{ padding: "10px 14px", borderRadius: 7, background: "rgba(251,191,36,0.06)", border: "1px solid rgba(251,191,36,0.2)" }}>
          <div style={{ fontSize: 9, color: "#fbbf24", fontWeight: 600, letterSpacing: 1.5, marginBottom: 5, textTransform: "uppercase" }}>5-Fold cross-validation</div>
          <div style={{ fontSize: 11, color: "rgba(255,255,255,0.55)", lineHeight: 1.7 }}>
            The transposed <code style={{ color: "#fbbf24" }}>[{nTrials} &times; {nVoxels}]</code> matrix is split into {nFolds} folds ({foldSize} trials each).
            For each fold: train the SVM on the other {nTrials - foldSize} rows, then predict the held-out {foldSize} rows.
            Mean accuracy across folds = one ROI accuracy value. Repeated 100&times; with random shuffling in the real experiment.
          </div>
        </div>
      </div>

      {/* Fold selector + legend */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
        <span style={{ fontSize: 10, color: "rgba(255,255,255,0.35)" }}>Show fold:</span>
        {Array.from({ length: nFolds }, (_, i) => (
          <button key={i} onClick={() => setFold(i)} style={{
            padding: "3px 10px", fontSize: 10, fontFamily: "inherit", borderRadius: 4, cursor: "pointer",
            border: `1px solid ${fold === i ? "rgba(251,191,36,0.5)" : "rgba(255,255,255,0.1)"}`,
            background: fold === i ? "rgba(251,191,36,0.12)" : "transparent",
            color: fold === i ? "#fbbf24" : "rgba(255,255,255,0.35)",
          }}>F{i + 1}</button>
        ))}
        <span style={{ fontSize: 9, color: "rgba(255,255,255,0.2)", marginLeft: 6 }}>highlighted on right matrix</span>
        <div style={{ marginLeft: "auto", display: "flex", gap: 12, flexWrap: "wrap" }}>
          {[comp.a, comp.b].map((c, i) => (
            <div key={c} style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, background: condColors[c] + "60", border: `1px solid ${condColors[c]}` }} />
              <span style={{ fontSize: 10, color: "rgba(255,255,255,0.4)" }}>{c} ({i === 0 ? "+1" : "\u22121"})</span>
            </div>
          ))}
          <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, border: "1.5px solid #fbbf24", background: "rgba(251,191,36,0.08)" }} />
            <span style={{ fontSize: 10, color: "rgba(255,255,255,0.4)" }}>test fold</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <div style={{ width: 22, height: 10, borderRadius: 2, background: "linear-gradient(to right,#3b82f6,#8b5cf6,#ef4444)" }} />
            <span style={{ fontSize: 10, color: "rgba(255,255,255,0.4)" }}>&beta; (blue=\u2212, red=+)</span>
          </div>
        </div>
      </div>

      <canvas ref={canvasRef} style={{ width, height: H, display: "block", maxWidth: "100%" }} />
    </div>
  );
}

// ─── SVM CLASSIFICATION (STEP 5) ────────────────────────────────────────────

function SVMClassification({ width = 1500 }) {
  const [comparison, setComparison] = useState("PlNt");
  const { allTrials, conditions } = SIMULATED_DATA;

  const comparisons = {
    PlNt: { a: "Pleasant", b: "Neutral", label: "Pleasant vs Neutral" },
    UpNt: { a: "Unpleasant", b: "Neutral", label: "Unpleasant vs Neutral" },
    PlUp: { a: "Pleasant", b: "Unpleasant", label: "Pleasant vs Unpleasant" },
  };

  const cvResult = useMemo(() => {
    const comp = comparisons[comparison];
    const dataA = allTrials.filter(t => t.condition === comp.a).map(t => t.betas);
    const dataB = allTrials.filter(t => t.condition === comp.b).map(t => t.betas);
    return runCrossValidation(dataA, dataB, 5);
  }, [comparison]);

  // 2D projection canvas (first 2 voxels)
  const scatterRef = useRef(null);
  const scatterH = 280;

  useEffect(() => {
    const canvas = scatterRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = scatterH * dpr; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, scatterH);

    const comp = comparisons[comparison];
    const dataA = allTrials.filter(t => t.condition === comp.a);
    const dataB = allTrials.filter(t => t.condition === comp.b);

    const pad = { top: 30, right: 30, bottom: 50, left: 65 };
    const plotW = width - pad.left - pad.right, plotH = scatterH - pad.top - pad.bottom;

    // Use voxel 1 vs voxel 3 for a nice separation
    const allX = [...dataA, ...dataB].map(t => t.betas[0]);
    const allY = [...dataA, ...dataB].map(t => t.betas[2]);
    const xMin = Math.min(...allX) - 3, xMax = Math.max(...allX) + 3;
    const yMin = Math.min(...allY) - 3, yMax = Math.max(...allY) + 3;
    const xScale = (v) => pad.left + ((v - xMin) / (xMax - xMin)) * plotW;
    const yScale = (v) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    // Grid
    ctx.strokeStyle = "rgba(255,255,255,0.05)"; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const xv = xMin + (i / 4) * (xMax - xMin), yv = yMin + (i / 4) * (yMax - yMin);
      ctx.beginPath(); ctx.moveTo(xScale(xv), pad.top); ctx.lineTo(xScale(xv), pad.top + plotH); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pad.left, yScale(yv)); ctx.lineTo(pad.left + plotW, yScale(yv)); ctx.stroke();
    }

    // Decision boundary (approximate - find SVM on 2D projection)
    const proj2A = dataA.map(t => [t.betas[0], t.betas[2]]);
    const proj2B = dataB.map(t => [t.betas[0], t.betas[2]]);
    const allProj = [...proj2A.map(p => ({ x: p, y: 1 })), ...proj2B.map(p => ({ x: p, y: -1 }))];
    const { normalized: normProj, means: pm, stds: ps } = zScore(allProj.map(d => d.x));
    const m2 = trainLinearSVM(normProj, allProj.map(d => d.y), 1.0, 0.01, 500);

    // Draw boundary region
    const step = 1;
    for (let px = pad.left; px < pad.left + plotW; px += step * 4) {
      for (let py = pad.top; py < pad.top + plotH; py += step * 4) {
        const fv0 = xMin + ((px - pad.left) / plotW) * (xMax - xMin);
        const fv1 = yMin + ((pad.top + plotH - py) / plotH) * (yMax - yMin);
        const nv0 = (fv0 - pm[0]) / ps[0], nv1 = (fv1 - pm[1]) / ps[1];
        const score = nv0 * m2.w[0] + nv1 * m2.w[1] + m2.b;
        if (Math.abs(score) < 0.8) {
          ctx.fillStyle = "rgba(255,255,255,0.03)";
          ctx.fillRect(px, py, 4, 4);
        }
      }
    }

    // Draw boundary line
    const linePoints = [];
    for (let xv = xMin; xv <= xMax; xv += 0.5) {
      const nv0 = (xv - pm[0]) / ps[0];
      const nv1 = -(m2.w[0] * nv0 + m2.b) / m2.w[1];
      const yv = nv1 * ps[1] + pm[1];
      if (yv >= yMin && yv <= yMax) linePoints.push([xScale(xv), yScale(yv)]);
    }
    if (linePoints.length > 1) {
      ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
      ctx.beginPath(); linePoints.forEach(([x, y], i) => { if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Points
    const drawPoints = (data, color) => {
      data.forEach(t => {
        const px = xScale(t.betas[0]), py = yScale(t.betas[2]);
        ctx.fillStyle = color + "30"; ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = color; ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#0c0a1a"; ctx.beginPath(); ctx.arc(px, py, 1.5, 0, Math.PI * 2); ctx.fill();
      });
    };
    drawPoints(dataB, condColors[comp.b]);
    drawPoints(dataA, condColors[comp.a]);

    // Axes
    ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top + plotH); ctx.lineTo(pad.left + plotW, pad.top + plotH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + plotH); ctx.stroke();

    ctx.fillStyle = "rgba(255,255,255,0.5)"; ctx.font = "11px 'DM Mono', monospace"; ctx.textAlign = "center";
    ctx.fillText("\u03B2 at Voxel 1", pad.left + plotW / 2, pad.top + plotH + 35);
    ctx.save(); ctx.translate(14, pad.top + plotH / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("\u03B2 at Voxel 3", 0, 0); ctx.restore();

    // Legend
    ctx.textAlign = "left"; ctx.font = "bold 11px 'DM Mono', monospace";
    ctx.fillStyle = condColors[comp.a]; ctx.fillText(`\u25CF ${comp.a}`, pad.left + 8, pad.top + 6);
    ctx.fillStyle = condColors[comp.b]; ctx.fillText(`\u25CF ${comp.b}`, pad.left + 8 + 120, pad.top + 6);
    ctx.fillStyle = "rgba(255,255,255,0.35)"; ctx.font = "10px 'DM Mono', monospace";
    ctx.fillText("--- SVM boundary", pad.left + 8 + 240, pad.top + 6);
  }, [comparison, width]);

  // Bar chart for fold accuracies
  const barRef = useRef(null);
  const barH = 140;
  useEffect(() => {
    const canvas = barRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = barH * dpr; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, barH);

    const pad = { top: 20, right: 30, bottom: 35, left: 65 };
    const plotW = width - pad.left - pad.right, plotH = barH - pad.top - pad.bottom;
    const folds = cvResult.foldAccuracies;
    const nFolds = folds.length;
    const barW = plotW / (nFolds * 2 + 1);
    const yMin2 = 0.3, yMax2 = 1.0;
    const yScale = (v) => pad.top + plotH - ((v - yMin2) / (yMax2 - yMin2)) * plotH;

    // Chance line
    ctx.strokeStyle = "rgba(255,255,255,0.15)"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(pad.left, yScale(0.5)); ctx.lineTo(pad.left + plotW, yScale(0.5)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "rgba(255,255,255,0.25)"; ctx.font = "9px 'DM Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText("chance = 50%", pad.left + plotW - 80, yScale(0.5) + 12);

    // Bars
    folds.forEach((acc, i) => {
      const x = pad.left + (i * 2 + 1) * barW;
      const h = yScale(0.5) - yScale(acc);
      const comp = comparisons[comparison];
      const grad = ctx.createLinearGradient(x, yScale(acc), x, yScale(0.5));
      grad.addColorStop(0, condColors[comp.a] + "90"); grad.addColorStop(1, condColors[comp.a] + "20");
      ctx.fillStyle = grad;
      ctx.fillRect(x, Math.min(yScale(0.5), yScale(acc)), barW, Math.abs(h));

      ctx.fillStyle = "rgba(255,255,255,0.7)"; ctx.font = "bold 10px 'DM Mono', monospace"; ctx.textAlign = "center";
      ctx.fillText(`${(acc * 100).toFixed(0)}%`, x + barW / 2, yScale(acc) - 6);
      ctx.fillStyle = "rgba(255,255,255,0.35)"; ctx.font = "9px 'DM Mono', monospace";
      ctx.fillText(`F${i + 1}`, x + barW / 2, pad.top + plotH + 14);
    });

    // Mean line
    ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(pad.left, yScale(cvResult.meanAccuracy)); ctx.lineTo(pad.left + plotW, yScale(cvResult.meanAccuracy)); ctx.stroke();
    ctx.fillStyle = "#fbbf24"; ctx.font = "bold 11px 'DM Mono', monospace"; ctx.textAlign = "right";
    ctx.fillText(`mean = ${(cvResult.meanAccuracy * 100).toFixed(1)}%`, pad.left + plotW, yScale(cvResult.meanAccuracy) - 6);

    // Y axis
    ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + plotH); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.4)"; ctx.font = "10px 'DM Mono', monospace"; ctx.textAlign = "right";
    [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].forEach(v => { if (v >= yMin2 && v <= yMax2) ctx.fillText(`${(v * 100).toFixed(0)}%`, pad.left - 6, yScale(v) + 3); });
  }, [cvResult, comparison, width]);

  const comp = comparisons[comparison];

  return (
    <div>
      <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 17, fontWeight: 600, margin: "0 0 4px" }}>SVM Classification</h2>
      <p style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", margin: "0 0 16px" }}>
        Step 5: Train linear SVM on &beta;&#770; patterns &middot; 5-fold cross-validation &middot; 20 trials per class
      </p>

      {/* Comparison selector */}
      <div style={{ display: "flex", gap: 6, marginBottom: 16 }}>
        {Object.entries(comparisons).map(([key, val]) => (
          <button key={key} onClick={() => setComparison(key)} style={{
            padding: "6px 14px", fontSize: 11, fontFamily: "inherit", borderRadius: 5, cursor: "pointer",
            border: `1px solid ${comparison === key ? "rgba(234,179,8,0.4)" : "rgba(255,255,255,0.08)"}`,
            background: comparison === key ? "rgba(234,179,8,0.12)" : "transparent",
            color: comparison === key ? "#fbbf24" : "rgba(255,255,255,0.35)",
          }}>{val.label}</button>
        ))}
      </div>

      {/* Feature matrix + CV split */}
      <SVMMatrix comparison={comparison} comparisons={comparisons} width={width} />

      {/* Scatter plot */}
      <div style={{ marginBottom: 4 }}>
        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 6, fontWeight: 500 }}>
          Feature Space (2D projection: Voxel 1 vs Voxel 3)
        </div>
        <canvas ref={scatterRef} style={{ width, height: scatterH, maxWidth: "100%" }} />
      </div>

      {/* Cross-validation bars */}
      <div style={{ marginTop: 12 }}>
        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 6, fontWeight: 500 }}>
          5-Fold Cross-Validation Accuracy
        </div>
        <canvas ref={barRef} style={{ width, height: barH, maxWidth: "100%" }} />
      </div>

      {/* Results summary */}
      <div style={{
        marginTop: 16, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 10,
      }}>
        <div style={{ padding: "12px", borderRadius: 8, textAlign: "center", background: "rgba(34,197,94,0.08)", border: "1px solid rgba(34,197,94,0.2)" }}>
          <div style={{ fontSize: 9, color: "rgba(255,255,255,0.4)", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>Mean Accuracy</div>
          <div style={{ fontSize: 22, fontWeight: 700, fontFamily: "'Space Grotesk', sans-serif", color: cvResult.meanAccuracy > 0.5 ? "#22c55e" : "#ef4444" }}>
            {(cvResult.meanAccuracy * 100).toFixed(1)}%
          </div>
        </div>
        <div style={{ padding: "12px", borderRadius: 8, textAlign: "center", background: "rgba(129,140,248,0.08)", border: "1px solid rgba(129,140,248,0.2)" }}>
          <div style={{ fontSize: 9, color: "rgba(255,255,255,0.4)", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>Chance Level</div>
          <div style={{ fontSize: 22, fontWeight: 700, fontFamily: "'Space Grotesk', sans-serif", color: "rgba(255,255,255,0.4)" }}>50.0%</div>
        </div>
        <div style={{ padding: "12px", borderRadius: 8, textAlign: "center", background: cvResult.meanAccuracy > 0.54 ? "rgba(234,179,8,0.08)" : "rgba(239,68,68,0.08)", border: `1px solid ${cvResult.meanAccuracy > 0.54 ? "rgba(234,179,8,0.2)" : "rgba(239,68,68,0.2)"}` }}>
          <div style={{ fontSize: 9, color: "rgba(255,255,255,0.4)", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>Significant?</div>
          <div style={{ fontSize: 22, fontWeight: 700, fontFamily: "'Space Grotesk', sans-serif", color: cvResult.meanAccuracy > 0.54 ? "#fbbf24" : "#ef4444" }}>
            {cvResult.meanAccuracy > 0.54 ? "YES \u2713" : "NO"}
          </div>
        </div>
      </div>

      <div style={{ marginTop: 16, padding: 16, background: "rgba(34,197,94,0.06)", border: "1px solid rgba(34,197,94,0.15)", borderRadius: 8, fontSize: 12, lineHeight: 1.7, color: "rgba(255,255,255,0.55)" }}>
        <span style={{ color: "#22c55e", fontWeight: 500 }}>Result &rarr; </span>
        The SVM achieves <strong>{(cvResult.meanAccuracy * 100).toFixed(1)}%</strong> accuracy for {comp.label}, well above the 50% chance level and the ~54% significance threshold (p &lt; 0.001 from permutation testing).
        This means V1v carries information that distinguishes {comp.a.toLowerCase()} from {comp.b.toLowerCase()} spatial activity patterns.
      </div>
    </div>
  );
}

// ─── PERMUTATION TEST (STEP 7) ───────────────────────────────────────────────

function PermutationTest({ width = 1500 }) {
  const [nPerms, setNPerms] = useState(10000);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const canvasRef = useRef(null);
  const animRef = useRef(null);

  // Simulate 20 subjects' accuracies per ROI (matching paper's Table)
  const observedAccuracies = useMemo(() => {
    const rng = mulberry32(777);
    const rois = ["V1v","V1d","V2v","V2d","V3v","V3d","hV4","VO1","VO2","PHC1","PHC2","hMT","LO1","LO2","V3a","V3b","IPS"];
    const meanAccs = [0.621,0.605,0.632,0.618,0.641,0.635,0.658,0.662,0.675,0.643,0.639,0.648,0.651,0.663,0.655,0.647,0.671];
    return rois.map((roi, ri) => {
      const subjectAccs = Array.from({ length: 20 }, () => meanAccs[ri] + gaussianRng(rng) * 0.042);
      const mean = subjectAccs.reduce((a, b) => a + b, 0) / 20;
      return { roi, subjectAccs, mean };
    });
  }, []);

  const [selectedROI, setSelectedROI] = useState(0);

  // Run permutation test
  const runPermutation = useCallback(() => {
    setIsRunning(true); setProgress(0); setResult(null);
    const roiData = observedAccuracies[selectedROI];
    const nSubjects = 20;
    const observedMean = roiData.mean;
    const rng = mulberry32(42 + selectedROI);

    // We simulate: for each permutation, randomly flip each subject's accuracy around 0.5
    // (equivalent to randomly swapping class labels → accuracy becomes 1-acc with prob 0.5)
    const nullMeans = [];
    const batchSize = 500;
    let done = 0;

    function processBatch() {
      const end = Math.min(done + batchSize, nPerms);
      for (let p = done; p < end; p++) {
        let sum = 0;
        for (let s = 0; s < nSubjects; s++) {
          const acc = roiData.subjectAccs[s];
          // Flip around 0.5: either keep acc or use (1 - acc)
          sum += rng() < 0.5 ? acc : (1 - acc);
        }
        nullMeans.push(sum / nSubjects);
      }
      done = end;
      setProgress(done / nPerms);

      if (done < nPerms) {
        animRef.current = requestAnimationFrame(processBatch);
      } else {
        // Compute p-value
        const nExceed = nullMeans.filter(m => m >= observedMean).length;
        const pValue = nExceed / nPerms;
        // Find threshold at p < 0.001
        const sorted = [...nullMeans].sort((a, b) => b - a);
        const threshIdx = Math.floor(nPerms * 0.001);
        const threshold = sorted[threshIdx] || sorted[0];

        setResult({ nullMeans, observedMean, pValue, nExceed, threshold });
        setIsRunning(false);
      }
    }
    animRef.current = requestAnimationFrame(processBatch);
  }, [nPerms, selectedROI, observedAccuracies]);

  useEffect(() => { return () => { if (animRef.current) cancelAnimationFrame(animRef.current); }; }, []);

  // Auto-run on mount/change
  useEffect(() => { runPermutation(); }, [selectedROI, nPerms]);

  // Draw histogram
  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas || !result) return;
    const ctx = canvas.getContext("2d"); const dpr = window.devicePixelRatio || 1;
    const histH = 300;
    canvas.width = width * dpr; canvas.height = histH * dpr; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, histH);

    const pad = { top: 30, right: 30, bottom: 55, left: 65 };
    const plotW = width - pad.left - pad.right, plotH = histH - pad.top - pad.bottom;

    const { nullMeans, observedMean, threshold } = result;

    // Bin the null distribution
    const nBins = 60;
    const binMin = Math.min(...nullMeans, 0.44);
    const binMax = Math.max(...nullMeans, observedMean + 0.02);
    const binW = (binMax - binMin) / nBins;
    const bins = Array(nBins).fill(0);
    nullMeans.forEach(v => {
      const idx = Math.min(nBins - 1, Math.max(0, Math.floor((v - binMin) / binW)));
      bins[idx]++;
    });
    const maxCount = Math.max(...bins);

    const xScale = (v) => pad.left + ((v - binMin) / (binMax - binMin)) * plotW;
    const yScale = (c) => pad.top + plotH - (c / maxCount) * plotH;
    const barPx = plotW / nBins;

    // Draw bars
    bins.forEach((count, i) => {
      const x = pad.left + i * barPx;
      const binCenter = binMin + (i + 0.5) * binW;
      const isAboveThresh = binCenter >= threshold;
      const isAboveObs = binCenter >= observedMean;

      const grad = ctx.createLinearGradient(x, yScale(count), x, yScale(0));
      if (isAboveObs) {
        grad.addColorStop(0, "rgba(239,68,68,0.7)"); grad.addColorStop(1, "rgba(239,68,68,0.15)");
      } else if (isAboveThresh) {
        grad.addColorStop(0, "rgba(234,179,8,0.5)"); grad.addColorStop(1, "rgba(234,179,8,0.1)");
      } else {
        grad.addColorStop(0, "rgba(99,102,241,0.5)"); grad.addColorStop(1, "rgba(99,102,241,0.08)");
      }
      ctx.fillStyle = grad;
      ctx.fillRect(x + 0.5, yScale(count), barPx - 1, yScale(0) - yScale(count));
    });

    // Threshold line (p < 0.001)
    const threshX = xScale(threshold);
    ctx.strokeStyle = "#eab308"; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(threshX, pad.top); ctx.lineTo(threshX, pad.top + plotH); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#eab308"; ctx.font = "bold 10px 'DM Mono', monospace"; ctx.textAlign = "center";
    ctx.fillText(`p<0.001 threshold`, threshX, pad.top - 5);
    ctx.fillText(`${(threshold * 100).toFixed(1)}%`, threshX, pad.top + 8);

    // Observed mean line
    const obsX = xScale(observedMean);
    ctx.strokeStyle = "#22c55e"; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(obsX, pad.top); ctx.lineTo(obsX, pad.top + plotH); ctx.stroke();
    // Arrow head
    ctx.fillStyle = "#22c55e";
    ctx.beginPath(); ctx.moveTo(obsX - 6, pad.top + 16); ctx.lineTo(obsX + 6, pad.top + 16); ctx.lineTo(obsX, pad.top + 24); ctx.fill();
    ctx.font = "bold 11px 'DM Mono', monospace";
    ctx.fillText(`observed = ${(observedMean * 100).toFixed(1)}%`, obsX, pad.top - 5);

    // Axes
    ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top + plotH); ctx.lineTo(pad.left + plotW, pad.top + plotH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + plotH); ctx.stroke();

    // X axis labels
    ctx.fillStyle = "rgba(255,255,255,0.5)"; ctx.font = "10px 'DM Mono', monospace"; ctx.textAlign = "center";
    const xStep = 0.02;
    for (let v = Math.ceil(binMin / xStep) * xStep; v <= binMax; v += xStep) {
      ctx.fillText(`${(v * 100).toFixed(0)}%`, xScale(v), pad.top + plotH + 16);
    }
    ctx.fillText("Group Mean Accuracy", pad.left + plotW / 2, pad.top + plotH + 40);

    // Y axis
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const c = Math.round((i / 4) * maxCount);
      ctx.fillText(`${c}`, pad.left - 8, yScale(c) + 3);
    }
    ctx.save(); ctx.translate(14, pad.top + plotH / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center";
    ctx.fillText("Count", 0, 0); ctx.restore();

    // Title
    ctx.fillStyle = "rgba(255,255,255,0.3)"; ctx.font = "10px 'DM Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText(`Null Distribution (${nPerms.toLocaleString()} permutations)`, pad.left, pad.top - 16);

  }, [result, width]);

  return (
    <div>
      <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 17, fontWeight: 600, margin: "0 0 4px" }}>Group-Level Permutation Test</h2>
      <p style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", margin: "0 0 16px" }}>
        Step 7: Test whether observed decoding accuracy across 20 subjects significantly exceeds chance (50%)
      </p>

      {/* How it works */}
      <div style={{ padding: "12px 16px", marginBottom: 16, borderRadius: 8, background: "rgba(99,102,241,0.06)", border: "1px solid rgba(99,102,241,0.15)", fontSize: 11, lineHeight: 1.8, color: "rgba(255,255,255,0.5)" }}>
        <span style={{ color: "#818cf8", fontWeight: 500 }}>How it works: </span>
        For each of {nPerms.toLocaleString()} permutations, randomly flip each subject's accuracy around 50% (simulating chance-level decoding). This builds a null distribution of group means. If the observed mean falls far into the tail, the decoding is significant.
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 10, marginBottom: 16, alignItems: "center", flexWrap: "wrap" }}>
        <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)" }}>ROI:</span>
        <select value={selectedROI} onChange={e => setSelectedROI(Number(e.target.value))} style={{
          padding: "5px 10px", fontSize: 11, fontFamily: "'DM Mono', monospace",
          background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.12)",
          color: "#c7d2fe", borderRadius: 5, cursor: "pointer",
        }}>
          {observedAccuracies.map((r, i) => <option key={i} value={i} style={{ background: "#1a1730" }}>{r.roi} (mean={((r.mean) * 100).toFixed(1)}%)</option>)}
        </select>

        <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginLeft: 8 }}>Permutations:</span>
        {[1000, 10000, 50000].map(n => (
          <button key={n} onClick={() => setNPerms(n)} style={{
            padding: "4px 10px", fontSize: 11, fontFamily: "inherit", borderRadius: 4, cursor: "pointer",
            border: `1px solid ${nPerms === n ? "rgba(129,140,248,0.4)" : "rgba(255,255,255,0.08)"}`,
            background: nPerms === n ? "rgba(129,140,248,0.12)" : "transparent",
            color: nPerms === n ? "#c7d2fe" : "rgba(255,255,255,0.3)",
          }}>{n >= 1000 ? `${n / 1000}k` : n}</button>
        ))}
      </div>

      {/* Progress bar when running */}
      {isRunning && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ height: 4, borderRadius: 2, background: "rgba(255,255,255,0.06)", overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${progress * 100}%`, background: "linear-gradient(90deg, #6366f1, #818cf8)", borderRadius: 2, transition: "width 0.1s" }} />
          </div>
          <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", marginTop: 4 }}>Running permutations... {Math.round(progress * 100)}%</div>
        </div>
      )}

      {/* Histogram */}
      <canvas ref={canvasRef} style={{ width, height: 300, maxWidth: "100%" }} />

      {/* Results cards */}
      {result && (
        <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 8 }}>
          <div style={{ padding: "10px", borderRadius: 8, textAlign: "center", background: "rgba(34,197,94,0.08)", border: "1px solid rgba(34,197,94,0.2)" }}>
            <div style={{ fontSize: 9, color: "rgba(255,255,255,0.4)", marginBottom: 3, textTransform: "uppercase", letterSpacing: 0.5 }}>Observed</div>
            <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "'Space Grotesk', sans-serif", color: "#22c55e" }}>{(result.observedMean * 100).toFixed(1)}%</div>
          </div>
          <div style={{ padding: "10px", borderRadius: 8, textAlign: "center", background: "rgba(234,179,8,0.08)", border: "1px solid rgba(234,179,8,0.2)" }}>
            <div style={{ fontSize: 9, color: "rgba(255,255,255,0.4)", marginBottom: 3, textTransform: "uppercase", letterSpacing: 0.5 }}>Threshold</div>
            <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "'Space Grotesk', sans-serif", color: "#eab308" }}>{(result.threshold * 100).toFixed(1)}%</div>
          </div>
          <div style={{ padding: "10px", borderRadius: 8, textAlign: "center", background: "rgba(129,140,248,0.08)", border: "1px solid rgba(129,140,248,0.2)" }}>
            <div style={{ fontSize: 9, color: "rgba(255,255,255,0.4)", marginBottom: 3, textTransform: "uppercase", letterSpacing: 0.5 }}>p-value</div>
            <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "'Space Grotesk', sans-serif", color: result.pValue < 0.001 ? "#818cf8" : "#ef4444" }}>
              {result.pValue === 0 ? `<${(1 / nPerms).toFixed(5)}` : result.pValue < 0.001 ? "<0.001" : result.pValue.toFixed(4)}
            </div>
          </div>
          <div style={{ padding: "10px", borderRadius: 8, textAlign: "center", background: result.pValue < 0.001 ? "rgba(34,197,94,0.08)" : "rgba(239,68,68,0.08)", border: `1px solid ${result.pValue < 0.001 ? "rgba(34,197,94,0.2)" : "rgba(239,68,68,0.2)"}` }}>
            <div style={{ fontSize: 9, color: "rgba(255,255,255,0.4)", marginBottom: 3, textTransform: "uppercase", letterSpacing: 0.5 }}>Significant?</div>
            <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "'Space Grotesk', sans-serif", color: result.pValue < 0.001 ? "#22c55e" : "#ef4444" }}>
              {result.pValue < 0.001 ? "YES \u2713" : "NO"}
            </div>
          </div>
        </div>
      )}

      {/* Supplementary t-test */}
      {result && (() => {
        const roiData = observedAccuracies[selectedROI];
        const mean = roiData.mean;
        const std = Math.sqrt(roiData.subjectAccs.reduce((s, a) => s + (a - mean) ** 2, 0) / 19);
        const se = std / Math.sqrt(20);
        const tStat = (mean - 0.5) / se;
        const cohenD = (mean - 0.5) / std;
        return (
          <div style={{ marginTop: 16, padding: 16, background: "rgba(129,140,248,0.06)", border: "1px solid rgba(129,140,248,0.15)", borderRadius: 8, fontSize: 12, lineHeight: 1.8, color: "rgba(255,255,255,0.55)" }}>
            <div style={{ color: "#818cf8", fontWeight: 500, fontSize: 11, textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>Supplementary one-sample t-test</div>
            <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, lineHeight: 2 }}>
              <span style={{ color: "rgba(255,255,255,0.4)" }}>H&#8320;: &mu; = 0.50 &nbsp;&nbsp; H&#8321;: &mu; &gt; 0.50</span><br />
              <span>mean = {(mean * 100).toFixed(1)}% &nbsp;&nbsp; std = {(std * 100).toFixed(1)}% &nbsp;&nbsp; n = 20</span><br />
              <span style={{ color: "#c7d2fe" }}>t = ({(mean * 100).toFixed(1)} &minus; 50) / ({(std * 100).toFixed(1)} / &radic;20) = <strong>{tStat.toFixed(2)}</strong></span><br />
              <span>Cohen's d = {cohenD.toFixed(2)} &nbsp;&nbsp; <strong style={{ color: "#22c55e" }}>p &lt; 0.0001 &rarr; SIGNIFICANT &check;</strong></span>
            </div>
          </div>
        );
      })()}
    </div>
  );
}

// ─── CONCLUSION / FINAL RESULTS (STEP 8) ────────────────────────────────────

function Conclusion({ width = 1500 }) {
  const roiResults = useMemo(() => {
    const rng = mulberry32(999);
    const rois = [
      { name: "V1v", region: "V1" }, { name: "V1d", region: "V1" },
      { name: "V2v", region: "V2" }, { name: "V2d", region: "V2" },
      { name: "V3v", region: "V3" }, { name: "V3d", region: "V3" },
      { name: "hV4", region: "V4" }, { name: "VO1", region: "VO" }, { name: "VO2", region: "VO" },
      { name: "PHC1", region: "PHC" }, { name: "PHC2", region: "PHC" },
      { name: "hMT", region: "MT" }, { name: "LO1", region: "LO" }, { name: "LO2", region: "LO" },
      { name: "V3a", region: "V3ab" }, { name: "V3b", region: "V3ab" },
      { name: "IPS", region: "IPS" },
    ];
    const plNtMeans = [62.1,60.5,63.2,61.8,64.1,63.5,65.8,66.2,67.5,64.3,63.9,64.8,65.1,66.3,65.5,64.7,67.1];
    const upNtMeans = [65.3,63.8,66.1,64.5,67.2,65.9,68.4,69.1,70.3,67.8,66.5,67.1,68.2,69.5,68.7,67.9,70.8];
    return rois.map((roi, i) => ({
      ...roi,
      plNt: plNtMeans[i],
      plNtSE: 0.8 + gaussianRng(rng) * 0.15,
      upNt: upNtMeans[i],
      upNtSE: 0.8 + gaussianRng(rng) * 0.2,
      sig: true,
    }));
  }, []);

  // Bar chart canvas
  const barRef = useRef(null);
  const barH = 320;
  useEffect(() => {
    const canvas = barRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = barH * dpr; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, barH);

    const pad = { top: 30, right: 20, bottom: 70, left: 55 };
    const plotW = width - pad.left - pad.right, plotH = barH - pad.top - pad.bottom;
    const n = roiResults.length;
    const groupW = plotW / n;
    const barW = groupW * 0.35;
    const yMin2 = 50, yMax2 = 75;
    const yScale = (v) => pad.top + plotH - ((v - yMin2) / (yMax2 - yMin2)) * plotH;

    // Grid
    ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.lineWidth = 1;
    for (let v = 50; v <= 75; v += 5) { ctx.beginPath(); ctx.moveTo(pad.left, yScale(v)); ctx.lineTo(pad.left + plotW, yScale(v)); ctx.stroke(); }

    // Chance line
    ctx.strokeStyle = "rgba(255,255,255,0.2)"; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(pad.left, yScale(50)); ctx.lineTo(pad.left + plotW, yScale(50)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(255,255,255,0.25)"; ctx.font = "9px 'DM Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText("chance = 50%", pad.left + 2, yScale(50) + 12);

    // Significance threshold
    ctx.strokeStyle = "rgba(234,179,8,0.4)"; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, yScale(54)); ctx.lineTo(pad.left + plotW, yScale(54)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(234,179,8,0.4)"; ctx.fillText("p<0.001 \u2248 54%", pad.left + 2, yScale(54) - 4);

    // Bars
    roiResults.forEach((roi, i) => {
      const cx = pad.left + (i + 0.5) * groupW;

      // Pl vs Nt bar
      const x1 = cx - barW - 1;
      const grad1 = ctx.createLinearGradient(x1, yScale(roi.plNt), x1, yScale(50));
      grad1.addColorStop(0, "rgba(34,197,94,0.7)"); grad1.addColorStop(1, "rgba(34,197,94,0.15)");
      ctx.fillStyle = grad1;
      ctx.fillRect(x1, yScale(roi.plNt), barW, yScale(50) - yScale(roi.plNt));

      // Up vs Nt bar
      const x2 = cx + 1;
      const grad2 = ctx.createLinearGradient(x2, yScale(roi.upNt), x2, yScale(50));
      grad2.addColorStop(0, "rgba(239,68,68,0.7)"); grad2.addColorStop(1, "rgba(239,68,68,0.15)");
      ctx.fillStyle = grad2;
      ctx.fillRect(x2, yScale(roi.upNt), barW, yScale(50) - yScale(roi.upNt));

      // Error bars (SE)
      [{ x: x1 + barW / 2, val: roi.plNt, se: roi.plNtSE, col: "#22c55e" },
       { x: x2 + barW / 2, val: roi.upNt, se: roi.upNtSE, col: "#ef4444" }].forEach(({ x, val, se, col }) => {
        ctx.strokeStyle = col + "80"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(x, yScale(val + se)); ctx.lineTo(x, yScale(val - se)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x - 3, yScale(val + se)); ctx.lineTo(x + 3, yScale(val + se)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x - 3, yScale(val - se)); ctx.lineTo(x + 3, yScale(val - se)); ctx.stroke();
      });

      // Significance stars
      ctx.fillStyle = "#fbbf24"; ctx.font = "bold 10px 'DM Mono', monospace"; ctx.textAlign = "center";
      ctx.fillText("***", cx, yScale(Math.max(roi.plNt, roi.upNt) + roi.upNtSE + 0.8));

      // ROI label
      ctx.save();
      ctx.translate(cx, pad.top + plotH + 12);
      ctx.rotate(-Math.PI / 4);
      ctx.fillStyle = "rgba(255,255,255,0.5)"; ctx.font = "10px 'DM Mono', monospace"; ctx.textAlign = "right";
      ctx.fillText(roi.name, 0, 0);
      ctx.restore();
    });

    // Y axis
    ctx.fillStyle = "rgba(255,255,255,0.4)"; ctx.font = "10px 'DM Mono', monospace"; ctx.textAlign = "right";
    for (let v = 50; v <= 75; v += 5) ctx.fillText(`${v}%`, pad.left - 6, yScale(v) + 3);

    ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + plotH); ctx.stroke();

    // Legend
    ctx.textAlign = "left"; ctx.font = "bold 11px 'DM Mono', monospace";
    const legX = pad.left + plotW - 200, legY = pad.top + 6;
    ctx.fillStyle = "#22c55e50"; ctx.fillRect(legX, legY - 6, 12, 12);
    ctx.fillStyle = "#22c55e"; ctx.fillText("Pl vs Nt", legX + 18, legY + 4);
    ctx.fillStyle = "#ef444450"; ctx.fillRect(legX + 100, legY - 6, 12, 12);
    ctx.fillStyle = "#ef4444"; ctx.fillText("Up vs Nt", legX + 118, legY + 4);
  }, [roiResults, width]);

  return (
    <div>
      <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 17, fontWeight: 600, margin: "0 0 4px" }}>Final Results</h2>
      <p style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", margin: "0 0 16px" }}>
        Step 8: Decoding accuracy across all 17 retinotopic ROIs &middot; 20 subjects &middot; p &lt; 0.001 (permutation)
      </p>

      {/* Chart */}
      <canvas ref={barRef} style={{ width, height: barH, maxWidth: "100%" }} />

      {/* Results table */}
      <div style={{ marginTop: 16, overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "'DM Mono', monospace" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
              <th style={{ textAlign: "left", padding: "8px 6px", color: "rgba(255,255,255,0.4)", fontWeight: 500 }}>ROI</th>
              <th style={{ textAlign: "center", padding: "8px 6px", color: "#22c55e", fontWeight: 500 }}>Pl vs Nt</th>
              <th style={{ textAlign: "center", padding: "8px 6px", color: "rgba(255,255,255,0.3)", fontWeight: 500 }}>p-value</th>
              <th style={{ textAlign: "center", padding: "8px 6px", color: "#ef4444", fontWeight: 500 }}>Up vs Nt</th>
              <th style={{ textAlign: "center", padding: "8px 6px", color: "rgba(255,255,255,0.3)", fontWeight: 500 }}>p-value</th>
              <th style={{ textAlign: "center", padding: "8px 6px", color: "#fbbf24", fontWeight: 500 }}>Sig</th>
            </tr>
          </thead>
          <tbody>
            {roiResults.map((roi, i) => (
              <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)", background: i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)" }}>
                <td style={{ padding: "6px", color: "rgba(255,255,255,0.7)", fontWeight: 500 }}>{roi.name}</td>
                <td style={{ padding: "6px", textAlign: "center", color: "#22c55e" }}>{roi.plNt.toFixed(1)} &plusmn; {Math.abs(roi.plNtSE).toFixed(1)}%</td>
                <td style={{ padding: "6px", textAlign: "center", color: "rgba(255,255,255,0.35)" }}>&lt; 0.001</td>
                <td style={{ padding: "6px", textAlign: "center", color: "#ef4444" }}>{roi.upNt.toFixed(1)} &plusmn; {Math.abs(roi.upNtSE).toFixed(1)}%</td>
                <td style={{ padding: "6px", textAlign: "center", color: "rgba(255,255,255,0.35)" }}>&lt; 0.001</td>
                <td style={{ padding: "6px", textAlign: "center", color: "#fbbf24" }}>***</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ fontSize: 10, color: "rgba(255,255,255,0.25)", marginTop: 6 }}>
          Chance level: 50% &nbsp;&middot;&nbsp; *** p &lt; 0.001 (permutation test)&nbsp;&middot;&nbsp; Values: mean &plusmn; SE across 20 subjects
        </div>
      </div>

      {/* Key finding */}
      <div style={{
        marginTop: 20, padding: 20, borderRadius: 10,
        background: "linear-gradient(135deg, rgba(34,197,94,0.08), rgba(129,140,248,0.06), rgba(239,68,68,0.05))",
        border: "1px solid rgba(34,197,94,0.2)",
      }}>
        <div style={{ fontSize: 13, fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, color: "#e2e0f0", marginBottom: 10 }}>
          Conclusion
        </div>
        <div style={{ fontSize: 12, lineHeight: 1.8, color: "rgba(255,255,255,0.6)" }}>
          <strong style={{ color: "#22c55e" }}>All 17 retinotopic ROIs</strong> show significant above-chance decoding for <em>both</em> comparisons (Pleasant vs. Neutral and Unpleasant vs. Neutral), with accuracies in the <strong>58&ndash;72%</strong> range.
          <br /><br />
          This includes <strong style={{ color: "#818cf8" }}>primary visual cortex (V1)</strong> &mdash; a region traditionally thought to only process low-level visual features. The result demonstrates that emotion-specific information is encoded in multi-voxel activity patterns even at the earliest stages of cortical visual processing.
          <br /><br />
          <span style={{ color: "rgba(255,255,255,0.4)", fontSize: 11 }}>Higher-order areas (VO2, IPS) show the strongest decoding, consistent with increasing abstractness along the visual hierarchy.</span>
        </div>
      </div>

      {/* Pipeline summary */}
      <div style={{ marginTop: 16, padding: 14, borderRadius: 8, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
        <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>Complete Pipeline Summary</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, fontSize: 10 }}>
          {[
            { label: "BOLD signal", color: "#f97316" },
            { label: "\u2192 HRF convolution", color: "#818cf8" },
            { label: "\u2192 Design matrix X", color: "#c7d2fe" },
            { label: "\u2192 GLM: solve for \u03B2", color: "#f472b6" },
            { label: "\u2192 Condition matrices", color: "#eab308" },
            { label: "\u2192 ROI mask", color: "#fbbf24" },
            { label: "\u2192 SVM (5-fold CV)", color: "#06b6d4" },
            { label: "\u2192 Permutation test", color: "#a855f7" },
            { label: "\u2192 Significant decoding \u2713", color: "#22c55e" },
          ].map(({ label, color }, i) => (
            <span key={i} style={{ padding: "3px 8px", borderRadius: 3, background: `${color}10`, border: `1px solid ${color}25`, color: `${color}cc` }}>{label}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── RAW BOLD MATRIX ─────────────────────────────────────────────────────────

function BoldDataMatrix({ width = 1500 }) {
  const canvasRef = useRef(null);
  const H = 420;
  const vKeys = ["v1","v2","v3","v4"];
  const vRGB  = [[249,115,22],[6,182,212],[168,85,247],[234,179,8]];
  const vLbls = ["V1","V2","V3","V4"];
  const vCols = [voxelColors.v1, voxelColors.v2, voxelColors.v3, voxelColors.v4];

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = H * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, H);

    const pad = { top: 40, right: 20, bottom: 24, left: 52 };
    const nR = 206, nC = 4;
    const cellW = (width - pad.left - pad.right) / nC;
    const cellH = (H - pad.top - pad.bottom) / nR;

    const colMin = vKeys.map(v => Math.min(...boldData.map(d => d[v])));
    const colMax = vKeys.map(v => Math.max(...boldData.map(d => d[v])));

    boldData.forEach((row, r) => {
      vKeys.forEach((v, c) => {
        const intensity = (row[v] - colMin[c]) / Math.max(colMax[c] - colMin[c], 0.001);
        const [R,G,B] = vRGB[c];
        ctx.fillStyle = `rgb(${Math.round(lerp(18,R,intensity))},${Math.round(lerp(12,G,intensity))},${Math.round(lerp(28,B,intensity))})`;
        ctx.fillRect(pad.left + c*cellW + 0.5, pad.top + r*cellH + 0.5, cellW-1, Math.max(cellH-0.5, 0.5));
      });
    });

    vLbls.forEach((lbl, c) => {
      ctx.fillStyle = vCols[c]; ctx.font = "bold 11px 'DM Mono',monospace"; ctx.textAlign = "center";
      ctx.fillText(lbl, pad.left + c*cellW + cellW/2, pad.top - 12);
    });
    ctx.fillStyle = "rgba(255,255,255,0.2)"; ctx.font = "8px 'DM Mono',monospace";
    ctx.fillText("voxel column →", pad.left + 2*cellW, pad.top - 2);

    ctx.fillStyle = "rgba(255,255,255,0.35)"; ctx.font = "9px 'DM Mono',monospace"; ctx.textAlign = "right";
    for (let r = 0; r < nR; r++) {
      if (r === 0 || (r+1) % 20 === 0)
        ctx.fillText(`t=${r+1}`, pad.left-6, pad.top+r*cellH+cellH/2+3);
    }

    ctx.strokeStyle = "rgba(255,255,255,0.08)"; ctx.lineWidth = 1;
    ctx.strokeRect(pad.left, pad.top, nC*cellW, nR*cellH);
    ctx.setLineDash([3,3]);
    for (let c = 1; c < nC; c++) { ctx.beginPath(); ctx.moveTo(pad.left+c*cellW,pad.top); ctx.lineTo(pad.left+c*cellW,pad.top+nR*cellH); ctx.stroke(); }
    ctx.setLineDash([]);

    ctx.fillStyle = "rgba(255,255,255,0.25)"; ctx.font = "9px 'DM Mono',monospace"; ctx.textAlign = "left";
    ctx.fillText("y  [206 TRs \u00D7 4 voxels]  — color brightness = relative signal amplitude per voxel column", pad.left, H-6);
  }, [width]);

  return <canvas ref={canvasRef} style={{ width, height: H, maxWidth: "100%" }} />;
}

// ─── HRF REGRESSOR MATRIX ────────────────────────────────────────────────────

function HrfRegressorMatrix({ width = 1500 }) {
  const canvasRef = useRef(null);
  const H = 420;
  // 60 columns: 20 Pl + 20 Nt + 20 Up
  const condGroups = [
    { label: "Pl", rgb: [34,197,94],  color: "#22c55e", start: 0,  end: 20 },
    { label: "Nt", rgb: [99,102,241], color: "#6366f1", start: 20, end: 40 },
    { label: "Up", rgb: [239,68,68],  color: "#ef4444", start: 40, end: 60 },
  ];

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = H * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, H);

    const pad = { top: 52, right: 20, bottom: 24, left: 52 };
    const nR = 206, nC = 60;
    const cellW = (width - pad.left - pad.right) / nC;
    const cellH = (H - pad.top - pad.bottom) / nR;
    const gMin = -0.03, gMax = 0.58;

    trials.forEach((trial, c) => {
      const grp = condGroups.find(g => trial.condition === g.label + "leasant" || trial.condition === g.label + "eutral" || trial.condition.startsWith(g.label));
      const grpMatch = condGroups.find(g => trial.condition === g.label.replace("Pl","Pleasant").replace("Nt","Neutral").replace("Up","Unpleasant") || g.label === trial.condition.slice(0,2));
      const band = condGroups.find(g => g.start <= c && c < g.end);
      const [R,G,B] = band.rgb;
      allRegressors[trial.name].forEach((val, r) => {
        const t = Math.max(0, Math.min(1, (val - gMin) / (gMax - gMin)));
        ctx.fillStyle = `rgb(${Math.round(lerp(18,R,t))},${Math.round(lerp(12,G,t))},${Math.round(lerp(28,B,t))})`;
        ctx.fillRect(pad.left+c*cellW+0.5, pad.top+r*cellH+0.5, cellW-0.5, Math.max(cellH-0.5, 0.5));
      });
    });

    // Group headers
    condGroups.forEach(g => {
      const cx = pad.left + (g.start + g.end) / 2 * cellW;
      ctx.fillStyle = g.color; ctx.font = "bold 10px 'DM Mono',monospace"; ctx.textAlign = "center";
      ctx.fillText(`${g.label}1–${g.label}20`, cx, pad.top - 28);
      ctx.fillStyle = g.color + "50"; ctx.font = "8px 'DM Mono',monospace";
      ctx.fillText("(20 HRF-convolved regressors)", cx, pad.top - 16);
      // bracket
      ctx.strokeStyle = g.color + "60"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.left+g.start*cellW, pad.top-8); ctx.lineTo(pad.left+g.end*cellW, pad.top-8); ctx.stroke();
      // separator
      if (g.start > 0) {
        ctx.strokeStyle = "rgba(255,255,255,0.2)"; ctx.setLineDash([3,3]);
        ctx.beginPath(); ctx.moveTo(pad.left+g.start*cellW, pad.top); ctx.lineTo(pad.left+g.start*cellW, pad.top+nR*cellH); ctx.stroke(); ctx.setLineDash([]);
      }
    });

    // Row labels
    ctx.fillStyle = "rgba(255,255,255,0.35)"; ctx.font = "9px 'DM Mono',monospace"; ctx.textAlign = "right";
    for (let r = 0; r < nR; r++) {
      if (r === 0 || (r+1) % 20 === 0)
        ctx.fillText(`t=${r+1}`, pad.left-4, pad.top+r*cellH+cellH/2+3);
    }

    ctx.strokeStyle = "rgba(255,255,255,0.08)"; ctx.lineWidth = 1;
    ctx.strokeRect(pad.left, pad.top, nC*cellW, nR*cellH);

    ctx.fillStyle = "rgba(255,255,255,0.25)"; ctx.font = "9px 'DM Mono',monospace"; ctx.textAlign = "left";
    ctx.fillText("X_task  [206 TRs \u00D7 60 trials]  — each column = one trial onset convolved with HRF \u2192 becomes a task column in design matrix X", pad.left, H-6);
  }, [width]);

  return <canvas ref={canvasRef} style={{ width, height: H, maxWidth: "100%" }} />;
}

// ─── VARIABLE REFERENCE COMPONENT ───────────────────────────────────────────

function VarRef({ rows }) {
  return (
    <div style={{ marginTop: 24, padding: "14px 18px", borderRadius: 8, background: "rgba(255,255,255,0.025)", border: "1px solid rgba(255,255,255,0.08)" }}>
      <div style={{ fontSize: 10, letterSpacing: 2, textTransform: "uppercase", color: "rgba(255,255,255,0.3)", fontWeight: 600, marginBottom: 10 }}>Variable Reference</div>
      <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, lineHeight: 1.7 }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", color: "rgba(255,255,255,0.35)", fontWeight: 500, paddingBottom: 4, borderBottom: "1px solid rgba(255,255,255,0.07)", width: "12%" }}>Symbol</th>
            <th style={{ textAlign: "left", color: "rgba(255,255,255,0.35)", fontWeight: 500, paddingBottom: 4, borderBottom: "1px solid rgba(255,255,255,0.07)", width: "20%" }}>Shape</th>
            <th style={{ textAlign: "left", color: "rgba(255,255,255,0.35)", fontWeight: 500, paddingBottom: 4, borderBottom: "1px solid rgba(255,255,255,0.07)" }}>Description</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([sym, shape, desc], i) => (
            <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
              <td style={{ padding: "5px 8px 5px 0", fontFamily: "'DM Mono', monospace", color: "#c7d2fe", fontWeight: 600 }}>{sym}</td>
              <td style={{ padding: "5px 8px", fontFamily: "'DM Mono', monospace", color: "rgba(255,255,255,0.4)", fontSize: 10 }}>{shape}</td>
              <td style={{ padding: "5px 0", color: "rgba(255,255,255,0.55)", wordBreak: "break-word" }}>{desc}</td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </div>
  );
}

const X_PREVIEW_ROWS = buildDesignMatrix().slice(0, 5);

export default function App() {
  const [highlightedVoxel, setHighlightedVoxel] = useState(null);
  const [activeTab, setActiveTab] = useState("bold");
  const containerRef = useRef(null);
  const chartWidth = useContainerWidth(containerRef);

  const tabs = [
    { id: "bold",       label: "\u2460 BOLD",       step: "Raw Data" },
    { id: "hrf",        label: "\u2461 HRF",        step: "Step 1a" },
    { id: "design",     label: "\u2462 Design",     step: "Step 1b" },
    { id: "glm",        label: "\u2463 GLM",        step: "Step 2a" },
    { id: "mumford",    label: "\u2463b Mumford",   step: "Step 2b" },
    { id: "betamatrix", label: "\u2464a \u03B2-Matrix", step: "Step 4a" },
    { id: "roimask",    label: "\u2464b ROI Mask",  step: "Step 4b" },
    { id: "condition",  label: "\u2464c Conditions", step: "Step 4c" },
    { id: "svm",        label: "\u2465 SVM",        step: "Step 5" },
    { id: "permutation",label: "\u2466 Permutation",step: "Step 7" },
    { id: "conclusion", label: "\u2467 Results",    step: "Step 8" },
  ];

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(145deg, #0c0a1a 0%, #151028 40%, #0f0d1f 100%)", color: "#e2e0f0", fontFamily: "'DM Mono', 'JetBrains Mono', monospace", padding: "32px 24px" }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet" />

      <div style={{  margin: "0 auto 24px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#22c55e", boxShadow: "0 0 8px #22c55e88" }} />
          <span style={{ fontSize: 11, letterSpacing: 3, textTransform: "uppercase", color: "rgba(255,255,255,0.35)", fontWeight: 500 }}>fMRI MVPA Pipeline</span>
        </div>
        <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 26, fontWeight: 700, margin: "0 0 6px", background: "linear-gradient(90deg, #e2e0f0, #818cf8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          BOLD &rarr; GLM &rarr; SVM &rarr; Significance
        </h1>
        <p style={{ fontSize: 12, color: "rgba(255,255,255,0.4)", margin: 0 }}>Bo et al. (2021) &middot; Subject S1 &middot; V1v ROI (4 voxels) &middot; 20 trials/condition &middot; 5-fold CV</p>
      </div>

      {/* Tab bar */}
      <div style={{  margin: "0 auto 16px", display: "flex", gap: 3, flexWrap: "wrap" }}>
        {tabs.map((tab, idx) => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
            padding: "7px 12px", fontSize: 11, fontFamily: "inherit",
            border: `1px solid ${activeTab === tab.id ? "rgba(129,140,248,0.4)" : "rgba(255,255,255,0.06)"}`,
            background: activeTab === tab.id ? "rgba(129,140,248,0.12)" : "transparent",
            color: activeTab === tab.id ? "#c7d2fe" : "rgba(255,255,255,0.3)",
            borderRadius: 6, cursor: "pointer", fontWeight: activeTab === tab.id ? 500 : 400,
            display: "flex", flexDirection: "column", alignItems: "center", gap: 1, minWidth: 65,
          }}>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Content */}
      <div ref={containerRef} style={{ margin: "0 auto", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: 24, overflowX: "hidden" }}>
        {activeTab === "bold" && (
          <>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16, flexWrap: "wrap", gap: 10 }}>
              <div>
                <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 17, fontWeight: 600, margin: "0 0 4px" }}>Raw BOLD Time Series</h2>
                <p style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", margin: 0 }}>y = measured signal &middot; [206 TRs &times; 4 voxels] &middot; 60 trials (20 Pl + 20 Nt + 20 Up)</p>
              </div>
              <div style={{ display: "flex", gap: 6 }}>
                {["v1","v2","v3","v4"].map((v, i) => (
                  <button key={v} onClick={() => setHighlightedVoxel(highlightedVoxel === v ? null : v)} style={{ padding: "4px 10px", fontSize: 11, fontFamily: "inherit", border: `1px solid ${highlightedVoxel === v ? voxelColors[v] : "rgba(255,255,255,0.1)"}`, background: highlightedVoxel === v ? voxelColors[v] + "20" : "transparent", color: voxelColors[v], borderRadius: 4, cursor: "pointer" }}>V{i + 1}</button>
                ))}
              </div>
            </div>
            <BoldChart data={boldData} highlighted={highlightedVoxel} width={chartWidth} />
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", fontWeight: 500, marginBottom: 8 }}>
                Data matrix y &nbsp;&middot;&nbsp; same signal as the chart above, displayed as a heatmap
              </div>
              <BoldDataMatrix width={chartWidth} />
            </div>
            <div style={{ display: "flex", gap: 16, justifyContent: "center", marginTop: 12, flexWrap: "wrap" }}>
              {[{color:"#22c55e",label:"Pleasant (Pl1–Pl20, 20 trials)"},{color:"#6366f1",label:"Neutral (Nt1–Nt20, 20 trials)"},{color:"#ef4444",label:"Unpleasant (Up1–Up20, 20 trials)"}].map(c => (
                <div key={c.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <div style={{ width: 10, height: 2, background: c.color + "80", borderTop: `1px dashed ${c.color}` }} />
                  <span style={{ fontSize: 11, color: "rgba(255,255,255,0.5)" }}>{c.label}</span>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 12, padding: 14, background: "rgba(129,140,248,0.06)", border: "1px solid rgba(129,140,248,0.15)", borderRadius: 8, fontSize: 12, lineHeight: 1.6, color: "rgba(255,255,255,0.55)" }}>
              <span style={{ color: "#818cf8", fontWeight: 500 }}>Condition patterns (mean beta per voxel): </span>
              <strong style={{ color: "#22c55e" }}>Pleasant</strong>: V1,V3 &uarr; / V2,V4 &darr; &nbsp;&middot;&nbsp;
              <strong style={{ color: "#6366f1" }}>Neutral</strong>: all moderate &uarr; &nbsp;&middot;&nbsp;
              <strong style={{ color: "#ef4444" }}>Unpleasant</strong>: V1,V3 &uarr;&uarr; / V2,V4 &darr;&darr;
            </div>
            <VarRef rows={[
              ["y", "206 × 4", "Preprocessed BOLD signal matrix; rows = time points (TRs), columns = voxels (V1–V4); values in arbitrary signal intensity units (~1000 a.u.)"],
              ["t", "1 × 206", "TR index 1–206 (TR = 1.98 s); total run duration = 206 × 1.98 ≈ 407 s"],
              ["Pl1…Pl20", "onset TRs", "20 Pleasant trial onset times (TRs 2, 11, 20, … 173); each trial lasts 1.5152 s"],
              ["Nt1…Nt20", "onset TRs", "20 Neutral trial onset times (TRs 5, 14, 23, … 176)"],
              ["Up1…Up20", "onset TRs", "20 Unpleasant trial onset times (TRs 8, 17, 26, … 179)"],
            ]} />
          </>
        )}

        {activeTab === "hrf" && (
          <>
            <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 17, fontWeight: 600, margin: "0 0 4px" }}>Canonical HRF</h2>
            <p style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", margin: "0 0 16px" }}>Pre-defined double-gamma &middot; TR = 1.98s</p>
            <HrfChart width={chartWidth} />
            <div style={{ marginTop: 16, padding: 14, background: "rgba(129,140,248,0.06)", border: "1px solid rgba(129,140,248,0.15)", borderRadius: 8, fontSize: 12, lineHeight: 1.7, color: "rgba(255,255,255,0.55)" }}>
              Convolve each trial onset with this HRF to produce one column of the design matrix X. Peak response at ~6s means BOLD peaks 2&ndash;3 TRs after stimulus.
            </div>
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", fontWeight: 500, marginBottom: 8 }}>
                Convolved regressors &nbsp;&middot;&nbsp; one column per trial (60 total) — these become the 60 task columns of the design matrix X
              </div>
              <HrfRegressorMatrix width={chartWidth} />
            </div>

            <div style={{ marginTop: 16, padding: 14, background: "rgba(129,140,248,0.06)", border: "1px solid rgba(129,140,248,0.15)", borderRadius: 8, fontSize: 12, lineHeight: 1.7, color: "rgba(255,255,255,0.55)" }}>
              <strong style={{ color: "rgba(255,255,255,0.7)" }}>Trial Onset Times (Run 1)</strong>
              <div style={{ marginTop: 8, fontFamily: "'DM Mono', monospace", fontSize: 11 }}>
                <span style={{ color: "#22c55e" }}>Pleasant (Pl1–Pl20):</span> TRs 2, 11, 20, 29, 38, 47, 56, 65, 74, 83, 92, 101, 110, 119, 128, 137, 146, 155, 164, 173<br/>
                <span style={{ color: "#6366f1" }}>Neutral (Nt1–Nt20):</span> TRs 5, 14, 23, 32, 41, 50, 59, 68, 77, 86, 95, 104, 113, 122, 131, 140, 149, 158, 167, 176<br/>
                <span style={{ color: "#ef4444" }}>Unpleasant (Up1–Up20):</span> TRs 8, 17, 26, 35, 44, 53, 62, 71, 80, 89, 98, 107, 116, 125, 134, 143, 152, 161, 170, 179
              </div>
            </div>
            <VarRef rows={[
              ["h(τ)", "9-point vector", "Canonical double-gamma HRF sampled at TR intervals (τ = 0, 1.98, 3.96, … 15.84 s); peak ≈ 6 s"],
              ["*", "convolution", "Each trial onset is convolved with h(τ): x_k[t] = Σ h[lag] × δ[t − onset_k − lag]; produces one regressor per trial"],
              ["x_k", "206 × 1", "HRF-convolved regressor for trial k; nonzero only in a ~9-TR window after each onset"],
              ["X_task", "206 × 60", "Matrix of all 60 trial regressors stacked as columns (20 Pl + 20 Nt + 20 Up)"],
            ]} />
          </>
        )}

        {activeTab === "design" && (
          <>
            <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 17, fontWeight: 600, margin: "0 0 4px" }}>Design Matrix X</h2>
            <p style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", margin: "0 0 16px" }}>[206 &times; 67]: 60 task regressors + 6 motion + 1 constant</p>
            <DesignMatrix width={chartWidth} />
            <VarRef rows={[
              ["X", "206 × 67", "Full design matrix; rows = TRs, columns = regressors"],
              ["X[:,0:20]", "206 × 20", "Pleasant trial regressors (Pl1–Pl20); each column = HRF-convolved onset impulse (green)"],
              ["X[:,20:40]", "206 × 20", "Neutral trial regressors (Nt1–Nt20) (purple)"],
              ["X[:,40:60]", "206 × 20", "Unpleasant trial regressors (Up1–Up20) (red)"],
              ["X[:,60:66]", "206 × 6", "Motion nuisance regressors: translations (tX, tY, tZ) + rotations (rX, rY, rZ) in mm/rad"],
              ["X[:,66]", "206 × 1", "Constant regressor (all 1s); models the baseline mean signal"],
            ]} />
          </>
        )}

        {activeTab === "glm" && (
          <><GlmSolver width={chartWidth} />
          <VarRef rows={[
            ["y", "206 × 1", "Observed BOLD signal for one voxel (one column of the full y matrix)"],
            ["X", "206 × 67", "Design matrix (60 task + 6 motion + 1 constant); built once per run"],
            ["β̂", "67 × 1", "OLS beta estimates for one voxel: β̂ = (XᵀX + λI)⁻¹ Xᵀy; first 60 entries are single-trial amplitudes"],
            ["β̂_Pl", "20 × 1", "Beta coefficients for the 20 Pleasant regressors (averaged → β̄_Pl shown in display)"],
            ["β̂_Nt", "20 × 1", "Beta coefficients for the 20 Neutral regressors"],
            ["β̂_Up", "20 × 1", "Beta coefficients for the 20 Unpleasant regressors"],
            ["ŷ = Xβ̂", "206 × 1", "Model-fitted BOLD signal (pink line in chart above)"],
            ["ε = y − ŷ", "206 × 1", "Residuals; should be white noise if model is adequate (bar chart below fit)"],
          ]} /></>
        )}
        {activeTab === "mumford" && (
          <><MumfordMethod width={chartWidth} />
          <VarRef rows={[
            ["X_t", "206 × 3", "Per-trial design matrix for trial t: [col₀ | col₁ | const]"],
            ["col₀", "206 × 1", "HRF regressor for trial t (the trial of interest)"],
            ["col₁", "206 × 1", "Sum of HRF regressors for all 59 other trials (nuisance absorber)"],
            ["β̂_t", "3 × 1", "OLS solution for trial t: [β₀, β₁, β_const]; only β₀ is kept"],
            ["β₀", "scalar", "Estimated response amplitude for trial t at the current voxel — this is the MVPA feature"],
            ["B_mumford", "60 × 4", "After running 60 GLMs: matrix of β₀ values; rows = trials, columns = voxels"],
          ]} /></>
        )}
        {activeTab === "betamatrix" && (
          <><BetaMatrix width={chartWidth} />
          <VarRef rows={[
            ["B", "60 × 4", "Beta matrix: 60 Mumford β₀ estimates × 4 voxels; rows 1–20 = Pleasant, 21–40 = Neutral, 41–60 = Unpleasant"],
            ["B[i,:]", "1 × 4", "Spatial pattern (across voxels) for trial i — this is the feature vector fed to the SVM"],
            ["B[:,j]", "60 × 1", "Response time-series for voxel j across all 60 trials"],
          ]} /></>
        )}
        {activeTab === "roimask" && (
          <><RoiMask width={chartWidth} />
          <VarRef rows={[
            ["B_full", "60 × V", "Beta matrix for all V voxels in the full brain (V can be 10,000+; here V=12 for illustration)"],
            ["m", "1 × V", "Binary ROI mask vector; m[j]=1 if voxel j is inside the ROI (e.g., V1v), 0 otherwise"],
            ["B_roi", "60 × 4", "Masked beta matrix: B_full[:, m==1]; only 4 ROI voxels retained here"],
            ["ROI", "—", "Region of interest (e.g., V1v, hV4, LO1); defined by a NIfTI binary mask binarized from atlas parcellation"],
          ]} /></>
        )}
        {activeTab === "condition" && (
          <><ConditionMatrices width={chartWidth} />
          <VarRef rows={[
            ["Pl", "4 × 20", "Pleasant condition matrix: rows = voxels, columns = 20 trial beta estimates; entry [v,k] = β₀ of Pl trial k at voxel v"],
            ["Nt", "4 × 20", "Neutral condition matrix (rows = voxels, columns = 20 trials)"],
            ["Up", "4 × 20", "Unpleasant condition matrix (rows = voxels, columns = 20 trials)"],
            ["μ_Pl", "4 × 1", "Mean spatial pattern for Pleasant: column-wise mean of Pl; used to confirm condition discriminability"],
          ]} /></>
        )}
        {activeTab === "svm" && (
          <><SVMClassification width={chartWidth} />
          <VarRef rows={[
            ["X_svm", "N × 4", "Input matrix for SVM: N samples (trials from 2 conditions) × 4 voxel features; z-scored before training"],
            ["y_svm", "N × 1", "Binary class labels: +1 (e.g., Pleasant) or −1 (e.g., Neutral)"],
            ["w", "4 × 1", "SVM weight vector (decision boundary normal); trained by gradient descent with hinge loss"],
            ["b", "scalar", "SVM bias term; decision: ŷ = sign(Xw + b)"],
            ["CV folds", "5 folds", "5-fold cross-validation: train on 80% trials, test on 20%; repeated 100× in real pipeline"],
            ["accuracy", "scalar", "Fraction of correctly classified test trials per fold; mean over folds = final score"],
          ]} /></>
        )}
        {activeTab === "permutation" && (
          <><PermutationTest width={chartWidth} />
          <VarRef rows={[
            ["acc_obs", "scalar", "Observed mean decoding accuracy across subjects for one ROI (e.g., 0.62 = 62%)"],
            ["null_dist", "N_perm × 1", "Null distribution: N_perm=10,000 permuted group mean accuracies (trial labels shuffled each time)"],
            ["p-value", "scalar", "p = #{null > acc_obs} / N_perm; significant if p < 0.001 (threshold = 99.9th percentile of null)"],
            ["chance", "0.50", "Chance level for binary classification (2 equally balanced classes)"],
            ["threshold", "scalar", "99.9th percentile of null distribution; significance threshold for p < 0.001 (uncorrected)"],
          ]} /></>
        )}
        {activeTab === "conclusion" && (
          <><Conclusion width={chartWidth} />
          <VarRef rows={[
            ["acc_ROI", "17 × 1", "Mean decoding accuracy per ROI across 16 subjects; each entry = mean of 100-rep × 5-fold CV"],
            ["SE", "17 × 1", "Standard error of the mean across subjects: SE = std(acc_subjects) / √16"],
            ["d_Cohen", "scalar", "Effect size: d = (acc_mean − 0.5) / std(acc_subjects); measures practical significance"],
            ["ROIs", "17", "Retinotopic areas tested: V1v/d, V2v/d, V3v/d, hV4, VO1/2, PHC1/2, hMT, LO1/2, V3a/b, IPS"],
            ["p < 0.001", "threshold", "Significance criterion (permutation-based, one-sample t-test vs. chance 0.5)"],
          ]} /></>
        )}
      </div>

      <div style={{  margin: "16px auto 0", textAlign: "center", fontSize: 10, color: "rgba(255,255,255,0.2)", letterSpacing: 1 }}>
        Bo et al. (2021) pipeline &middot; Toy 4-voxel example with simulated multi-run data
      </div>
    </div>
  );
}
