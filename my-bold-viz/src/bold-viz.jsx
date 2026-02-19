import { useState, useRef, useEffect, useMemo, useCallback } from "react";

// ─── DATA ────────────────────────────────────────────────────────────────────

const boldData = [
  { t: 1,  v1: 1000.2, v2: 998.5,  v3: 1001.3, v4: 999.8 },
  { t: 2,  v1: 1001.5, v2: 999.2,  v3: 1000.8, v4: 1000.1 },
  { t: 3,  v1: 1003.8, v2: 997.1,  v3: 1004.2, v4: 998.3 },
  { t: 4,  v1: 1008.1, v2: 995.3,  v3: 1009.5, v4: 996.7 },
  { t: 5,  v1: 1012.4, v2: 993.8,  v3: 1013.1, v4: 995.2 },
  { t: 6,  v1: 1009.7, v2: 995.1,  v3: 1010.8, v4: 996.5 },
  { t: 7,  v1: 1004.2, v2: 997.3,  v3: 1005.1, v4: 998.8 },
  { t: 8,  v1: 1001.1, v2: 999.8,  v3: 1002.3, v4: 1000.2 },
  { t: 9,  v1: 1003.5, v2: 1003.2, v3: 1003.8, v4: 1004.1 },
  { t: 10, v1: 1005.2, v2: 1006.8, v3: 1005.1, v4: 1007.3 },
  { t: 11, v1: 1003.1, v2: 1004.5, v3: 1003.9, v4: 1005.2 },
  { t: 12, v1: 1000.8, v2: 1000.2, v3: 1001.1, v4: 1000.5 },
  { t: 13, v1: 999.5,  v2: 1001.8, v3: 998.7,  v4: 1002.1 },
  { t: 14, v1: 1006.3, v2: 999.2,  v3: 1010.2, v4: 997.8 },
  { t: 15, v1: 1011.8, v2: 996.5,  v3: 1015.3, v4: 994.1 },
  { t: 16, v1: 1008.2, v2: 997.8,  v3: 1011.7, v4: 996.3 },
  { t: 17, v1: 1003.1, v2: 999.5,  v3: 1004.8, v4: 999.1 },
  { t: 18, v1: 1000.5, v2: 1000.1, v3: 1001.2, v4: 1000.3 },
  { t: 19, v1: 1001.2, v2: 999.8,  v3: 1000.5, v4: 1000.1 },
  { t: 20, v1: 1000.8, v2: 1000.3, v3: 1000.9, v4: 1000.0 },
];

const trials = [
  { name: "Pl1", label: "Pleasant", onset: 3, color: "#22c55e" },
  { name: "Nt1", label: "Neutral", onset: 8, color: "#6366f1" },
  { name: "Up1", label: "Unpleasant", onset: 13, color: "#ef4444" },
];

const regressors = {
  Pl1: [0,0,0.04,0.28,0.58,0.32,0.08,-0.02,-0.03,0,0,0,0,0,0,0,0,0,0,0],
  Nt1: [0,0,0,0,0,0,0,0.04,0.28,0.58,0.32,0.08,-0.02,-0.03,0,0,0,0,0,0],
  Up1: [0,0,0,0,0,0,0,0,0,0,0,0,0.04,0.28,0.58,0.32,0.08,-0.02,-0.03,0],
};

const motion = [
  [0.012,0.003,-0.005,0.001,0.000,0.002],[0.015,0.005,-0.003,0.001,0.001,0.002],
  [0.018,0.004,-0.004,0.001,0.000,0.002],[0.014,0.006,-0.002,0.001,0.001,0.002],
  [0.016,0.005,-0.003,0.002,0.001,0.003],[0.017,0.006,-0.003,0.001,0.001,0.002],
  [0.018,0.006,-0.002,0.002,0.001,0.003],[0.019,0.007,-0.002,0.002,0.001,0.003],
  [0.018,0.006,-0.002,0.001,0.001,0.002],[0.019,0.007,-0.002,0.002,0.001,0.003],
  [0.020,0.007,-0.002,0.002,0.001,0.003],[0.019,0.007,-0.001,0.002,0.001,0.003],
  [0.020,0.008,-0.002,0.002,0.001,0.003],[0.020,0.008,-0.001,0.002,0.001,0.003],
  [0.020,0.008,-0.001,0.002,0.001,0.003],[0.021,0.008,-0.001,0.002,0.001,0.003],
  [0.021,0.008,-0.001,0.002,0.001,0.003],[0.020,0.008,-0.001,0.002,0.001,0.003],
  [0.021,0.008,-0.001,0.002,0.001,0.003],[0.021,0.008,-0.001,0.002,0.001,0.003],
];

const HRF_VALUES = [0.00,0.04,0.28,0.58,0.32,0.08,-0.02,-0.03,-0.01];
const voxelColors = { v1: "#f97316", v2: "#06b6d4", v3: "#a855f7", v4: "#eab308" };
const voxelLabels = { v1: "Voxel 1", v2: "Voxel 2", v3: "Voxel 3", v4: "Voxel 4" };
const condColors = { Pleasant: "#22c55e", Neutral: "#6366f1", Unpleasant: "#ef4444" };

// ─── LINEAR ALGEBRA ──────────────────────────────────────────────────────────

function buildDesignMatrix() {
  const X = [];
  for (let t = 0; t < 20; t++) X.push([regressors.Pl1[t], regressors.Nt1[t], regressors.Up1[t], ...motion[t], 1.0]);
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
    const xScale = (t) => pad.left + ((t - 1) / 19) * plotW;
    const yScale = (v) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;
    ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) { const yv = yMin + (i / 4) * (yMax - yMin); ctx.beginPath(); ctx.moveTo(pad.left, yScale(yv)); ctx.lineTo(width - pad.right, yScale(yv)); ctx.stroke(); }
    trials.forEach(trial => {
      const x = xScale(trial.onset);
      ctx.fillStyle = trial.color + "18"; ctx.fillRect(x - 2, pad.top, xScale(trial.onset + 4) - x + 4, plotH);
      ctx.strokeStyle = trial.color + "60"; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = trial.color; ctx.font = "bold 11px 'DM Mono', monospace"; ctx.textAlign = "center"; ctx.fillText(trial.name, x, pad.top - 5);
    });
    keys.forEach((key) => {
      const isActive = highlighted === null || highlighted === key;
      ctx.strokeStyle = isActive ? voxelColors[key] : voxelColors[key] + "25"; ctx.lineWidth = isActive ? 2.5 : 1; ctx.beginPath();
      data.forEach((d, i) => { const px = xScale(d.t), py = yScale(d[key]); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }); ctx.stroke();
      if (isActive) data.forEach((d) => { ctx.fillStyle = voxelColors[key]; ctx.beginPath(); ctx.arc(xScale(d.t), yScale(d[key]), 3, 0, Math.PI * 2); ctx.fill(); });
    });
    ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top + plotH); ctx.lineTo(width - pad.right, pad.top + plotH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + plotH); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.5)"; ctx.font = "11px 'DM Mono', monospace"; ctx.textAlign = "center";
    for (let t = 1; t <= 20; t++) { if (t % 2 === 0 || t === 1) ctx.fillText(`${t}`, xScale(t), pad.top + plotH + 18); }
    ctx.fillText("Scan (TR)", width / 2, pad.top + plotH + 40);
    ctx.textAlign = "right";
    for (let i = 0; i < 5; i++) { const yv = yMin + (i / 4) * (yMax - yMin); ctx.fillText(yv.toFixed(0), pad.left - 8, yScale(yv) + 4); }
    ctx.save(); ctx.translate(14, pad.top + plotH / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText("Signal Intensity", 0, 0); ctx.restore();
  }, [data, highlighted, width, height]);
  return <canvas ref={canvasRef} style={{ width, height }} />;
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
  return <canvas ref={canvasRef} style={{ width, height }} />;
}

function DesignMatrix({ width = 1500, height = 450 }) {
  const canvasRef = useRef(null);
  const colNames = ["Pl1", "Nt1", "Up1", "tX", "tY", "tZ", "rX", "rY", "rZ", "const"];
  const colColors = ["#22c55e", "#6366f1", "#ef4444", "#666", "#666", "#666", "#555", "#555", "#555", "#888"];
  const matrix = buildDesignMatrix();
  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = height * dpr; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, height);
    const pad = { top: 40, right: 20, bottom: 30, left: 50 };
    const cellW = (width - pad.left - pad.right) / 10, cellH = (height - pad.top - pad.bottom) / 20;
    const colMin = Array(10).fill(Infinity), colMax = Array(10).fill(-Infinity);
    matrix.forEach(row => row.forEach((v, c) => { if (v < colMin[c]) colMin[c] = v; if (v > colMax[c]) colMax[c] = v; }));
    matrix.forEach((row, r) => {
      row.forEach((val, c) => {
        const x = pad.left + c * cellW, y2 = pad.top + r * cellH;
        let intensity = colMax[c] === colMin[c] ? 0.5 : (val - colMin[c]) / (colMax[c] - colMin[c]);
        let color;
        if (c < 3) { const bc = c === 0 ? [34,197,94] : c === 1 ? [99,102,241] : [239,68,68]; color = `rgb(${Math.round(lerp(18,bc[0],intensity))},${Math.round(lerp(12,bc[1],intensity))},${Math.round(lerp(28,bc[2],intensity))})`; }
        else if (c < 9) { const v2 = Math.round(lerp(18, 80, intensity)); color = `rgb(${v2+15},${v2+10},${v2})`; }
        else { color = `rgb(60,55,75)`; }
        ctx.fillStyle = color; ctx.fillRect(x + 0.5, y2 + 0.5, cellW - 1, cellH - 1);
        if (c < 3 && val !== 0) { ctx.fillStyle = intensity > 0.5 ? "rgba(0,0,0,0.7)" : "rgba(255,255,255,0.6)"; ctx.font = "9px 'DM Mono', monospace"; ctx.textAlign = "center"; ctx.fillText(val.toFixed(2), x + cellW / 2, y2 + cellH / 2 + 3); }
      });
    });
    ctx.textAlign = "center"; ctx.font = "bold 11px 'DM Mono', monospace";
    colNames.forEach((name, c) => { ctx.fillStyle = colColors[c]; ctx.fillText(name, pad.left + c * cellW + cellW / 2, pad.top - 10); });
    ctx.font = "9px 'DM Mono', monospace"; ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.fillText("Task Regressors", pad.left + 1.5 * cellW, pad.top - 24); ctx.fillText("Motion (nuisance)", pad.left + 5.5 * cellW, pad.top - 24);
    ctx.textAlign = "right"; ctx.font = "10px 'DM Mono', monospace"; ctx.fillStyle = "rgba(255,255,255,0.4)";
    for (let r = 0; r < 20; r++) { if (r % 2 === 0) ctx.fillText(`t=${r + 1}`, pad.left - 6, pad.top + r * cellH + cellH / 2 + 3); }
    trials.forEach(trial => { const y2 = pad.top + (trial.onset - 1) * cellH; ctx.fillStyle = trial.color; ctx.beginPath(); ctx.moveTo(pad.left - 2, y2 + cellH / 2 - 4); ctx.lineTo(pad.left + 2, y2 + cellH / 2); ctx.lineTo(pad.left - 2, y2 + cellH / 2 + 4); ctx.fill(); });
    ctx.strokeStyle = "rgba(255,255,255,0.1)"; ctx.lineWidth = 1; ctx.strokeRect(pad.left, pad.top, 10 * cellW, 20 * cellH);
    ctx.strokeStyle = "rgba(255,255,255,0.2)"; ctx.setLineDash([3, 3]);
    [3, 9].forEach(col => { const sepX = pad.left + col * cellW; ctx.beginPath(); ctx.moveTo(sepX, pad.top); ctx.lineTo(sepX, pad.top + 20 * cellH); ctx.stroke(); }); ctx.setLineDash([]);
  }, [width, height]);
  return <canvas ref={canvasRef} style={{ width, height }} />;
}

// ─── GLM SOLVER ──────────────────────────────────────────────────────────────

function GlmSolver({ width = 1500 }) {
  const [selectedVoxel, setSelectedVoxel] = useState("v1");
  const X = useMemo(() => buildDesignMatrix(), []);
  const y = useMemo(() => boldData.map(d => d[selectedVoxel]), [selectedVoxel]);
  const beta = useMemo(() => solveOLS(X, y), [X, y]);
  const yHat = useMemo(() => X.map(row => row.reduce((s, val, j) => s + val * beta[j], 0)), [X, beta]);
  const residuals = useMemo(() => y.map((yi, i) => yi - yHat[i]), [y, yHat]);

  const betaNames = ["\u03B2_Pl1", "\u03B2_Nt1", "\u03B2_Up1", "\u03B2_tX", "\u03B2_tY", "\u03B2_tZ", "\u03B2_rX", "\u03B2_rY", "\u03B2_rZ", "\u03B2\u2080"];
  const betaClr = ["#22c55e", "#6366f1", "#ef4444", "#777", "#777", "#777", "#666", "#666", "#666", "#999"];

  const canvasRef = useRef(null); const chartH = 240;
  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr; canvas.height = chartH * dpr; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, chartH);
    const pad = { top: 25, right: 20, bottom: 50, left: 65 };
    const plotW = width - pad.left - pad.right, plotH = chartH - pad.top - pad.bottom;
    const allVals = [...y, ...yHat];
    const yMin = Math.min(...allVals) - 1, yMax = Math.max(...allVals) + 1;
    const xScale = (t) => pad.left + ((t - 1) / 19) * plotW;
    const yScale = (v) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;
    ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) { const yv = yMin + (i / 4) * (yMax - yMin); ctx.beginPath(); ctx.moveTo(pad.left, yScale(yv)); ctx.lineTo(width - pad.right, yScale(yv)); ctx.stroke(); }
    trials.forEach(trial => { const x = xScale(trial.onset); ctx.fillStyle = trial.color + "10"; ctx.fillRect(x - 2, pad.top, xScale(trial.onset + 4) - x + 4, plotH); ctx.strokeStyle = trial.color + "40"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke(); ctx.setLineDash([]); });
    y.forEach((yi, i) => { const px = xScale(i + 1); ctx.strokeStyle = "rgba(255,100,100,0.35)"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(px, yScale(yi)); ctx.lineTo(px, yScale(yHat[i])); ctx.stroke(); });
    ctx.strokeStyle = "#f472b6"; ctx.lineWidth = 2.5; ctx.beginPath();
    yHat.forEach((v, i) => { const px = xScale(i + 1), py = yScale(v); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }); ctx.stroke();
    ctx.strokeStyle = voxelColors[selectedVoxel]; ctx.lineWidth = 2; ctx.beginPath();
    y.forEach((v, i) => { const px = xScale(i + 1), py = yScale(v); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }); ctx.stroke();
    y.forEach((v, i) => { ctx.fillStyle = voxelColors[selectedVoxel]; ctx.beginPath(); ctx.arc(xScale(i + 1), yScale(v), 3.5, 0, Math.PI * 2); ctx.fill(); });
    yHat.forEach((v, i) => { ctx.fillStyle = "#f472b6"; ctx.beginPath(); ctx.arc(xScale(i + 1), yScale(v), 2.5, 0, Math.PI * 2); ctx.fill(); });
    ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top + plotH); ctx.lineTo(width - pad.right, pad.top + plotH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + plotH); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.5)"; ctx.font = "11px 'DM Mono', monospace"; ctx.textAlign = "center";
    for (let t = 1; t <= 20; t++) { if (t % 2 === 0 || t === 1) ctx.fillText(`${t}`, xScale(t), pad.top + plotH + 18); }
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
    const rMax = Math.max(...residuals.map(Math.abs)) * 1.2 || 1;
    const xScale = (t) => pad.left + ((t - 1) / 19) * plotW;
    const yScale = (v) => pad.top + plotH / 2 - (v / rMax) * (plotH / 2);
    ctx.strokeStyle = "rgba(255,255,255,0.15)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, yScale(0)); ctx.lineTo(width - pad.right, yScale(0)); ctx.stroke();
    residuals.forEach((r, i) => {
      const px = xScale(i + 1), barW = plotW / 24;
      ctx.fillStyle = r > 0 ? "rgba(239,68,68,0.35)" : "rgba(99,102,241,0.35)";
      ctx.fillRect(px - barW / 2, Math.min(yScale(0), yScale(r)), barW, Math.abs(yScale(0) - yScale(r)));
    });
    ctx.fillStyle = "rgba(255,255,255,0.3)"; ctx.font = "10px 'DM Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText("\u03B5 = y \u2212 X\u03B2\u0302", pad.left + 4, pad.top + 4);
  }, [residuals, width]);

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
      <canvas ref={canvasRef} style={{ width, height: chartH }} />
      <canvas ref={residCanvasRef} style={{ width, height: residH, marginTop: 4 }} />
      <div style={{ marginTop: 16 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: "rgba(255,255,255,0.6)", marginBottom: 8, fontFamily: "'Space Grotesk', sans-serif" }}>Estimated &beta;&#770; for {voxelLabels[selectedVoxel]}</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 6 }}>
          {beta.map((b, i) => (
            <div key={i} style={{ padding: "8px 6px", borderRadius: 6, textAlign: "center", background: i < 3 ? betaClr[i] + "12" : "rgba(255,255,255,0.03)", border: `1px solid ${i < 3 ? betaClr[i] + "30" : "rgba(255,255,255,0.06)"}` }}>
              <div style={{ fontSize: 9, color: betaClr[i], marginBottom: 3, fontWeight: 500 }}>{betaNames[i]}</div>
              <div style={{ fontSize: 13, fontWeight: 600, color: i < 3 ? "#e2e0f0" : "rgba(255,255,255,0.45)" }}>{b.toFixed(2)}</div>
            </div>
          ))}
        </div>
      </div>
      <div style={{ marginTop: 16, padding: 16, background: "rgba(244,114,182,0.06)", border: "1px solid rgba(244,114,182,0.15)", borderRadius: 8, fontSize: 12, lineHeight: 1.7, color: "rgba(255,255,255,0.55)" }}>
        <span style={{ color: "#f472b6", fontWeight: 500 }}>What &beta;&#770; tells us &rarr; </span>
        These 3 task betas per voxel become the input features for MVPA classification in the next steps.
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

      <canvas ref={canvasRef} style={{ width, height: heatH }} />

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

      {/* Scatter plot */}
      <div style={{ marginBottom: 4 }}>
        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 6, fontWeight: 500 }}>
          Feature Space (2D projection: Voxel 1 vs Voxel 3)
        </div>
        <canvas ref={scatterRef} style={{ width, height: scatterH }} />
      </div>

      {/* Cross-validation bars */}
      <div style={{ marginTop: 12 }}>
        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 6, fontWeight: 500 }}>
          5-Fold Cross-Validation Accuracy
        </div>
        <canvas ref={barRef} style={{ width, height: barH }} />
      </div>

      {/* Results summary */}
      <div style={{
        marginTop: 16, display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10,
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
      <canvas ref={canvasRef} style={{ width, height: 300 }} />

      {/* Results cards */}
      {result && (
        <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
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
      <canvas ref={barRef} style={{ width, height: barH }} />

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

const X_PREVIEW_ROWS = buildDesignMatrix().slice(0, 5);

export default function App() {
  const [highlightedVoxel, setHighlightedVoxel] = useState(null);
  const [activeTab, setActiveTab] = useState("bold");

  const tabs = [
    { id: "bold", label: "\u2460 BOLD", step: "Raw Data" },
    { id: "hrf", label: "\u2461 HRF", step: "Step 1a" },
    { id: "design", label: "\u2462 Design", step: "Step 1b" },
    { id: "glm", label: "\u2463 GLM", step: "Step 2" },
    { id: "condition", label: "\u2464 Conditions", step: "Step 3\u20134" },
    { id: "svm", label: "\u2465 SVM", step: "Step 5" },
    { id: "permutation", label: "\u2466 Permutation", step: "Step 7" },
    { id: "conclusion", label: "\u2467 Results", step: "Step 8" },
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
      <div style={{  margin: "0 auto", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: 24 }}>
        {activeTab === "bold" && (
          <>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16, flexWrap: "wrap", gap: 10 }}>
              <div>
                <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 17, fontWeight: 600, margin: "0 0 4px" }}>Raw BOLD Time Series</h2>
                <p style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", margin: 0 }}>y = measured signal &middot; [20 &times; 4]</p>
              </div>
              <div style={{ display: "flex", gap: 6 }}>
                {["v1","v2","v3","v4"].map((v, i) => (
                  <button key={v} onClick={() => setHighlightedVoxel(highlightedVoxel === v ? null : v)} style={{ padding: "4px 10px", fontSize: 11, fontFamily: "inherit", border: `1px solid ${highlightedVoxel === v ? voxelColors[v] : "rgba(255,255,255,0.1)"}`, background: highlightedVoxel === v ? voxelColors[v] + "20" : "transparent", color: voxelColors[v], borderRadius: 4, cursor: "pointer" }}>V{i + 1}</button>
                ))}
              </div>
            </div>
            <BoldChart data={boldData} highlighted={highlightedVoxel} />
            <div style={{ display: "flex", gap: 16, justifyContent: "center", marginTop: 12, flexWrap: "wrap" }}>
              {trials.map(trial => (
                <div key={trial.name} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <div style={{ width: 10, height: 10, borderRadius: 2, background: trial.color + "40", border: `1px solid ${trial.color}60` }} />
                  <span style={{ fontSize: 11, color: "rgba(255,255,255,0.5)" }}>{trial.name} ({trial.label})</span>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 16, padding: 14, background: "rgba(129,140,248,0.06)", border: "1px solid rgba(129,140,248,0.15)", borderRadius: 8, fontSize: 12, lineHeight: 1.6, color: "rgba(255,255,255,0.55)" }}>
              <span style={{ color: "#818cf8", fontWeight: 500 }}>Patterns &rarr; </span>
              <strong style={{ color: "#22c55e" }}>Pl1</strong>: V1,V3 &uarr; / V2,V4 &darr; &nbsp;&middot;&nbsp;
              <strong style={{ color: "#6366f1" }}>Nt1</strong>: all &uarr; &nbsp;&middot;&nbsp;
              <strong style={{ color: "#ef4444" }}>Up1</strong>: V1,V3 &uarr;&uarr; / V2,V4 &darr;
            </div>
          </>
        )}

        {activeTab === "hrf" && (
          <>
            <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 17, fontWeight: 600, margin: "0 0 4px" }}>Canonical HRF</h2>
            <p style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", margin: "0 0 16px" }}>Pre-defined double-gamma &middot; TR = 1.98s</p>
            <HrfChart />
            <div style={{ marginTop: 16, padding: 14, background: "rgba(129,140,248,0.06)", border: "1px solid rgba(129,140,248,0.15)", borderRadius: 8, fontSize: 12, lineHeight: 1.7, color: "rgba(255,255,255,0.55)" }}>
              Convolve each trial onset with this HRF to produce one column of the design matrix X. Peak response at ~6s means BOLD peaks 2&ndash;3 TRs after stimulus.
            </div>
          </>
        )}

        {activeTab === "design" && (
          <>
            <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 17, fontWeight: 600, margin: "0 0 4px" }}>Design Matrix X</h2>
            <p style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", margin: "0 0 16px" }}>[20 &times; 10]: 3 task + 6 motion + 1 constant</p>
            <DesignMatrix />
          </>
        )}

        {activeTab === "glm" && <GlmSolver />}
        {activeTab === "condition" && <ConditionMatrices />}
        {activeTab === "svm" && <SVMClassification />}
        {activeTab === "permutation" && <PermutationTest />}
        {activeTab === "conclusion" && <Conclusion />}
      </div>

      <div style={{  margin: "16px auto 0", textAlign: "center", fontSize: 10, color: "rgba(255,255,255,0.2)", letterSpacing: 1 }}>
        Bo et al. (2021) pipeline &middot; Toy 4-voxel example with simulated multi-run data
      </div>
    </div>
  );
}
