// Deterministic Monte Carlo forecast for make.fun's pumpfun top-5 competition.
// Runs with Node 22 and the standard library only.

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MICRO = 1_000_000;

const clamp = (value: number, low: number, high: number): number => Math.max(low, Math.min(high, value));

function sizeOf(lamports: string | number): number {
  return Math.log10(Number(BigInt(lamports))) - 9;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) | 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  history_until: string;
  reads: string[];
  size: { edges: number[] };
};
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8').split('\n').filter((line) => line !== '');
const marks = lines
  .map((line) => JSON.parse(line) as { at: string; ok: boolean; rows: Array<Record<string, unknown>> | null })
  .filter((mark) => mark.ok === true && Array.isArray(mark.rows) && mark.rows.length > 0)
  .sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
if (marks.length === 0) throw new Error('the history holds no ok mark with rows');

const firstTime = Date.parse(marks[0]!.at);
const timeOf = (iso: string): number => (Date.parse(iso) - firstTime) / 3_600_000;
const horizon = Math.max(
  0.1,
  task.reads.reduce((sum, read) => sum + (Date.parse(read) - Date.parse(task.history_until)) / 3_600_000, 0) / task.reads.length,
);
const observationsByMint = new Map<string, Array<{ time: number; size: number; rank: number }>>();
const metadataByMint = new Map<string, { pool: boolean; createdAt: string | null }>();
const lastMark = marks[marks.length - 1]!;
const lastRows = (lastMark.rows as Array<Record<string, unknown>>).slice().sort((a, b) => Number(a.rank) - Number(b.rank));

for (const mark of marks) {
  const time = timeOf(mark.at);
  for (const row of mark.rows as Array<Record<string, unknown>>) {
    const mint = String(row.mint);
    const size = sizeOf(row.market_cap_lamports as string | number);
    const observations = observationsByMint.get(mint) ?? [];
    observations.push({ time, size, rank: Number(row.rank) });
    observationsByMint.set(mint, observations);
    const metadata = metadataByMint.get(mint) ?? { pool: false, createdAt: null };
    metadata.pool ||= row.venue === 'pool' || row.complete === true;
    if (typeof row.created_at === 'string' && (metadata.createdAt === null || row.created_at < metadata.createdAt)) {
      metadata.createdAt = row.created_at;
    }
    metadataByMint.set(mint, metadata);
  }
}

const smallestLastSize = Math.min(...lastRows.map((row) => sizeOf(row.market_cap_lamports as string | number)));
const lastMintSet = new Set(lastRows.map((row) => String(row.mint)));
const features = new Map<string, {
  mint: string;
  lastRank: number;
  lastY: number;
  slope: number;
  vol: number;
  pool: boolean;
  createdAt: string | null;
}>();

for (const [mint, observations] of observationsByMint) {
  const recent = observations.slice(-4);
  let slope = 0;
  if (recent.length >= 2) {
    const meanTime = recent.reduce((sum, observation) => sum + observation.time, 0) / recent.length;
    const meanSize = recent.reduce((sum, observation) => sum + observation.size, 0) / recent.length;
    const denominator = recent.reduce((sum, observation) => sum + (observation.time - meanTime) ** 2, 0);
    if (denominator > 0) {
      slope = recent.reduce((sum, observation) => sum + (observation.time - meanTime) * (observation.size - meanSize), 0) / denominator;
    }
  }
  const rates: number[] = [];
  for (let index = 1; index < observations.length; index++) {
    const previous = observations[index - 1]!;
    const current = observations[index]!;
    const dt = current.time - previous.time;
    if (dt > 0) rates.push((current.size - previous.size) / Math.sqrt(dt));
  }
  let vol = 0.3;
  if (observations.length >= 2 && rates.length > 0) {
    const meanRate = rates.reduce((sum, rate) => sum + rate, 0) / rates.length;
    const variance = rates.reduce((sum, rate) => sum + (rate - meanRate) ** 2, 0) / rates.length;
    vol = clamp(Math.sqrt(variance), 0.08, 1.2);
  }
  const lastObservation = observations[observations.length - 1]!;
  features.set(mint, {
    mint,
    lastRank: lastObservation.rank,
    lastY: lastObservation.size,
    slope,
    vol,
    pool: metadataByMint.get(mint)!.pool,
    createdAt: metadataByMint.get(mint)!.createdAt,
  });
}

const random = mulberry32(20261112);
const normal = (() => {
  let spare: number | null = null;
  return (): number => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    const u = Math.max(Number.MIN_VALUE, random());
    const v = random();
    const magnitude = Math.sqrt(-2 * Math.log(u));
    spare = magnitude * Math.sin(2 * Math.PI * v);
    return magnitude * Math.cos(2 * Math.PI * v);
  };
})();

const bestRankByMint = new Map<string, number>();
for (const [mint, observations] of observationsByMint) {
  bestRankByMint.set(mint, Math.min(...observations.map((observation) => observation.rank)));
}
const candidates = [...features.values()]
  .filter((coin) => lastMintSet.has(coin.mint) || (bestRankByMint.get(coin.mint) ?? Infinity) <= 12)
  .sort((a, b) => a.lastRank - b.lastRank || a.mint.localeCompare(b.mint))
  .slice(0, 220)
  .map((coin) => ({
    ...coin,
    lastY: lastMintSet.has(coin.mint) ? coin.lastY : Math.min(coin.lastY, smallestLastSize),
    mu: 0,
    sigma: 0,
    pCrash: 0,
  }));
if (candidates.length < task.places) throw new Error('the history holds fewer candidates than forecast places');

for (const coin of candidates) {
  const drift = clamp(0.5 * clamp(coin.slope, -0.9, 0.9) * Math.min(horizon, 2.5), -0.55, 0.5);
  coin.mu = coin.lastY + drift + (coin.pool ? 0.03 : 0);
  // Volatility growth flattens after a few hours.
  coin.sigma = clamp((0.3 + 0.55 * coin.vol) * Math.sqrt(Math.min(horizon, 4) + 0.25), 0.22, 1.35) * (coin.pool ? 0.92 : 1.08);
  coin.pCrash = clamp(
    (coin.pool ? 0.05 : 0.13) * (0.7 + 0.3 * Math.sqrt(horizon)) + (coin.slope < -0.2 ? 0.06 : 0),
    0.04,
    0.35,
  );
}

const SIMULATIONS = 9_000;
const counts = new Map(candidates.map((coin) => [coin.mint, new Uint32Array(task.places)]));
const placeSamples = Array.from({ length: task.places }, () => new Float64Array(SIMULATIONS));
const simulated = candidates.map((coin) => ({ coin, size: 0 }));
for (let simulation = 0; simulation < SIMULATIONS; simulation++) {
  for (const item of simulated) {
    const coin = item.coin;
    if (random() < coin.pCrash) {
      item.size = coin.mu - 0.9 - 1.3 * random();
    } else {
      const tailScale = random() < 0.18 ? 1.9 : 1;
      item.size = coin.mu + coin.sigma * normal() * tailScale;
    }
  }
  const ranking = simulated.slice().sort((a, b) => b.size - a.size || a.coin.mint.localeCompare(b.coin.mint));
  for (let placeIndex = 0; placeIndex < task.places; placeIndex++) {
    const winner = ranking[placeIndex]!;
    counts.get(winner.coin.mint)![placeIndex]!++;
    placeSamples[placeIndex]![simulation] = winner.size;
  }
}

const probabilities: Array<{ mint: string; place: number; p: number }> = [];
for (let place = 1; place <= task.places; place++) {
  const reserve = clamp(0.4 + 0.03 * (place - 1) + 0.07 * Math.sqrt(horizon), 0.4, 0.66);
  const rows = candidates
    .map((coin) => ({
      mint: coin.mint,
      place,
      p: Math.floor((counts.get(coin.mint)![place - 1]! / SIMULATIONS) * (1 - reserve) * MICRO) / MICRO,
    }))
    .filter((row) => row.p >= 0.0006)
    .sort((a, b) => b.p - a.p || a.mint.localeCompare(b.mint))
    .slice(0, 100);
  probabilities.push(...rows);
}

const chosen = new Set<string>();
const top5: string[] = [];
for (let placeIndex = 0; placeIndex < task.places; placeIndex++) {
  let bestMint: string | null = null;
  let bestCount = -1;
  for (const coin of candidates) {
    if (chosen.has(coin.mint)) continue;
    const count = counts.get(coin.mint)![placeIndex]!;
    if (count > bestCount || (count === bestCount && (bestMint === null || coin.mint < bestMint))) {
      bestCount = count;
      bestMint = coin.mint;
    }
  }
  if (bestMint !== null) {
    chosen.add(bestMint);
    top5.push(bestMint);
  }
}

/** erfc(x), Numerical Recipes' erfcc approximation. */
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const above = (x: number, mean: number, sd: number): number =>
  (x === -Infinity ? 1 : x === Infinity ? 0 : 0.5 * erfc((x - mean) / (sd * Math.SQRT2)));

const sizes: Array<{ place: number; bin: number; p: number }> = [];
const edges = task.size.edges;
for (let placeIndex = 0; placeIndex < task.places; placeIndex++) {
  const samples = placeSamples[placeIndex]!;
  let mean = 0;
  for (const sample of samples) mean += sample / SIMULATIONS;
  let variance = 0;
  for (const sample of samples) variance += (sample - mean) ** 2 / SIMULATIONS;
  const sd = Math.sqrt(variance) * 1.5 + 0.08;
  const empirical = new Uint32Array(edges.length + 1);
  for (const sample of samples) {
    let bin = 0;
    while (bin < edges.length && sample >= edges[bin]!) bin++;
    empirical[bin]!++;
  }
  for (let bin = 0; bin <= edges.length; bin++) {
    const lo = bin === 0 ? -Infinity : edges[bin - 1]!;
    const hi = bin === edges.length ? Infinity : edges[bin]!;
    const wideNormalShare = above(lo, mean, sd) - above(hi, mean, sd);
    const p = Math.floor((0.68 * (empirical[bin]! / SIMULATIONS) + 0.32 * wideNormalShare) * MICRO) / MICRO;
    if (p > 0) sizes.push({ place: placeIndex + 1, bin, p });
  }
}

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5, sizes })}\n`);
