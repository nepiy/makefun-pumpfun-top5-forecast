const fs = require("node:fs");
const path = require("node:path");

// 15000 simulations smooth the Monte Carlo noise in the place shares.
const SIMULATION_COUNT = 15000;
const SEED = 20261112;
const PLACES = 5;
const MIN_REPORTED_PROBABILITY = 0.0006;
const PROBABILITY_SCALE = 1000000;

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function parsedTime(value) {
  const milliseconds = Date.parse(value);
  return Number.isFinite(milliseconds) ? milliseconds : null;
}

function standardDeviation(values) {
  if (values.length === 0) return 0;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    values.length;
  return Math.sqrt(variance);
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return function random() {
    let value = (state += 0x6d2b79f5);
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function makeNormalGenerator(random) {
  let spare = null;
  return function normal() {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    const radius = Math.sqrt(-2 * Math.log(1 - random()));
    const angle = 2 * Math.PI * random();
    spare = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  };
}

function erfc(value) {
  const absolute = Math.abs(value);
  const t = 1 / (1 + 0.5 * absolute);
  const answer =
    t *
    Math.exp(
      -absolute * absolute -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t *
                          (0.27886807 +
                            t *
                              (-1.13520398 +
                                t *
                                  (1.48851587 +
                                    t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return value >= 0 ? answer : 2 - answer;
}

function normalCdfAt(value, mean, deviation) {
  const z = (value - mean) / (deviation * Math.SQRT2);
  return 0.5 * erfc(-z);
}

function floorProbability(value) {
  return Math.floor(value * PROBABILITY_SCALE) / PROBABILITY_SCALE;
}

function sizeToBin(value, edges) {
  let low = 0;
  let high = edges.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (value >= edges[middle]) low = middle + 1;
    else high = middle;
  }
  return low;
}

function readMarks(historyText) {
  const marks = [];
  for (const line of historyText.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const mark = JSON.parse(line);
      if (
        mark &&
        mark.ok === true &&
        Array.isArray(mark.rows) &&
        mark.rows.length > 0 &&
        typeof mark.at === "string"
      ) {
        marks.push(mark);
      }
    } catch {
      // Ignore malformed lines; only valid successful marks are observations.
    }
  }
  marks.sort((left, right) => left.at.localeCompare(right.at));
  return marks;
}

function horizonHours(task) {
  const historyTime = parsedTime(task.history_until);
  const readTimes = Array.isArray(task.reads)
    ? task.reads.map(parsedTime).filter((time) => time !== null)
    : [];
  if (historyTime === null || readTimes.length === 0) return 0.1;
  const meanDifference =
    readTimes.reduce((sum, time) => sum + (time - historyTime), 0) /
    readTimes.length /
    3600000;
  return Math.max(0.1, meanDifference);
}

function collectCoins(marks) {
  const coins = new Map();
  const lastMarkByMint = new Map();
  let lastSmallestSize = Infinity;
  const firstMarkTime = parsedTime(marks[0].at);

  for (let markIndex = 0; markIndex < marks.length; markIndex += 1) {
    const mark = marks[markIndex];
    const markTime = parsedTime(mark.at);
    if (markTime === null) continue;
    const timeHours =
      firstMarkTime === null ? 0 : (markTime - firstMarkTime) / 3600000;
    const seenThisMark = new Set();

    for (const row of mark.rows) {
      if (!row || typeof row.mint !== "string" || !row.mint) continue;
      let marketCap;
      try {
        marketCap = Number(BigInt(row.market_cap_lamports));
      } catch {
        continue;
      }
      if (!(marketCap > 0) || !Number.isFinite(marketCap)) continue;
      const size = Math.log10(marketCap) - 9;
      const rank = Number(row.rank);
      if (!Number.isFinite(size) || !Number.isFinite(rank)) continue;

      let coin = coins.get(row.mint);
      if (!coin) {
        coin = {
          mint: row.mint,
          observations: [],
          pool: false,
          createdAt: null,
          bestRank: Infinity,
          lastObservation: null,
        };
        coins.set(row.mint, coin);
      }
      coin.observations.push({ time: timeHours, size, rank });
      coin.bestRank = Math.min(coin.bestRank, rank);
      coin.pool = coin.pool || row.venue === "pool" || row.complete === true;
      if (coin.createdAt === null && typeof row.created_at === "string") {
        coin.createdAt = row.created_at;
      }
      coin.lastObservation = { time: timeHours, size, rank };

      if (markIndex === marks.length - 1) {
        lastSmallestSize = Math.min(lastSmallestSize, size);
        if (!seenThisMark.has(row.mint)) {
          const existing = lastMarkByMint.get(row.mint);
          if (!existing || rank < existing.rank) {
            lastMarkByMint.set(row.mint, { size, rank });
          }
          seenThisMark.add(row.mint);
        }
      }
    }
  }

  const candidates = [];
  for (const coin of coins.values()) {
    const lastMarkObservation = lastMarkByMint.get(coin.mint);
    if (lastMarkObservation) {
      candidates.push({
        ...coin,
        lastY: lastMarkObservation.size,
        lastRank: lastMarkObservation.rank,
      });
    } else if (coin.bestRank <= 12 && coin.lastObservation) {
      candidates.push({
        ...coin,
        lastY: Math.min(coin.lastObservation.size, lastSmallestSize),
        lastRank: coin.lastObservation.rank,
      });
    }
  }

  candidates.sort(
    (left, right) =>
      left.lastRank - right.lastRank || left.mint.localeCompare(right.mint),
  );
  return candidates.slice(0, 220);
}

function addSimulationParameters(candidates, horizon) {
  for (const coin of candidates) {
    const observations = coin.observations.slice(-4);
    let slope = 0;
    if (observations.length >= 2) {
      const meanTime =
        observations.reduce((sum, item) => sum + item.time, 0) /
        observations.length;
      const meanSize =
        observations.reduce((sum, item) => sum + item.size, 0) /
        observations.length;
      let numerator = 0;
      let denominator = 0;
      for (const observation of observations) {
        const deltaTime = observation.time - meanTime;
        numerator += deltaTime * (observation.size - meanSize);
        denominator += deltaTime * deltaTime;
      }
      if (denominator > 0) slope = numerator / denominator;
    }

    let volatility = 0.3;
    if (observations.length >= 2) {
      const increments = [];
      for (let index = 1; index < observations.length; index += 1) {
        const deltaTime =
          observations[index].time - observations[index - 1].time;
        if (deltaTime > 0) {
          increments.push(
            (observations[index].size - observations[index - 1].size) /
              Math.sqrt(deltaTime),
          );
        }
      }
      volatility = clamp(standardDeviation(increments), 0.08, 1.2);
    }

    const poolAdjustment = coin.pool ? 0.03 : 0;
    const drift = clamp(
      0.5 * clamp(slope, -0.9, 0.9) * Math.min(horizon, 2.5),
      -0.55,
      0.5,
    );
    const sigma =
      clamp(
        (0.3 + 0.55 * volatility) *
          Math.sqrt(Math.min(horizon, 4) + 0.25),
        0.22,
        1.35,
      ) * (coin.pool ? 0.92 : 1.08);
    const pCrash = clamp(
      (coin.pool ? 0.05 : 0.13) * (0.7 + 0.3 * Math.sqrt(horizon)) +
        (slope < -0.2 ? 0.06 : 0),
      0.04,
      0.35,
    );

    coin.slope = slope;
    coin.volatility = volatility;
    coin.mu = coin.lastY + drift + poolAdjustment;
    coin.sigma = sigma;
    coin.pCrash = pCrash;
  }
}

function simulate(candidates, horizon, edges, taskPlaces) {
  if (candidates.length < taskPlaces) {
    throw new Error("Not enough observed candidates to produce a distinct top5.");
  }
  addSimulationParameters(candidates, horizon);
  const random = mulberry32(SEED);
  const normal = makeNormalGenerator(random);
  const counts = candidates.map(() => new Uint32Array(taskPlaces));
  const placeSizes = Array.from(
    { length: taskPlaces },
    () => new Float64Array(SIMULATION_COUNT),
  );
  const sampledSizes = new Float64Array(candidates.length);
  const ranked = new Array(candidates.length);

  for (let simulation = 0; simulation < SIMULATION_COUNT; simulation += 1) {
    for (let index = 0; index < candidates.length; index += 1) {
      const coin = candidates[index];
      if (random() < coin.pCrash) {
        sampledSizes[index] = coin.mu - 0.9 - 1.3 * random();
      } else {
        let z = normal();
        if (random() < 0.18) z *= 1.9;
        sampledSizes[index] = coin.mu + coin.sigma * z;
      }
      ranked[index] = index;
    }

    ranked.sort(
      (left, right) =>
        sampledSizes[right] - sampledSizes[left] ||
        candidates[left].mint.localeCompare(candidates[right].mint),
    );
    for (let place = 0; place < taskPlaces; place += 1) {
      const candidateIndex = ranked[place];
      counts[candidateIndex][place] += 1;
      placeSizes[place][simulation] = sampledSizes[candidateIndex];
    }
  }

  const probabilities = [];
  for (let place = 0; place < taskPlaces; place += 1) {
    const reserve = clamp(
      0.4 + 0.03 * place + 0.07 * Math.sqrt(horizon),
      0.4,
      0.66,
    );
    const rows = [];
    for (let index = 0; index < candidates.length; index += 1) {
      const probability = floorProbability(
        (counts[index][place] / SIMULATION_COUNT) * (1 - reserve),
      );
      if (probability >= MIN_REPORTED_PROBABILITY) {
        rows.push({
          mint: candidates[index].mint,
          place: place + 1,
          p: probability,
        });
      }
    }
    rows.sort(
      (left, right) =>
        right.p - left.p || left.mint.localeCompare(right.mint),
    );
    probabilities.push(...rows.slice(0, 100));
  }

  const top5 = [];
  const used = new Set();
  for (let place = 0; place < taskPlaces; place += 1) {
    let bestIndex = -1;
    for (let index = 0; index < candidates.length; index += 1) {
      if (used.has(index)) continue;
      if (
        bestIndex === -1 ||
        counts[index][place] > counts[bestIndex][place] ||
        (counts[index][place] === counts[bestIndex][place] &&
          candidates[index].mint.localeCompare(candidates[bestIndex].mint) < 0)
      ) {
        bestIndex = index;
      }
    }
    used.add(bestIndex);
    top5.push(candidates[bestIndex].mint);
  }

  const sizes = [];
  for (let place = 0; place < taskPlaces; place += 1) {
    const values = placeSizes[place];
    const mean =
      values.reduce((sum, value) => sum + value, 0) / SIMULATION_COUNT;
    const variance =
      values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
      SIMULATION_COUNT;
    const wideDeviation = Math.sqrt(variance) * 1.5 + 0.08;
    const empiricalBins = new Uint32Array(edges.length + 1);
    for (const value of values) empiricalBins[sizeToBin(value, edges)] += 1;

    let lowerCdf = 0;
    for (let bin = 0; bin <= edges.length; bin += 1) {
      let wideShare;
      if (bin === 0) {
        const upperCdf = normalCdfAt(edges[0], mean, wideDeviation);
        wideShare = upperCdf;
        lowerCdf = upperCdf;
      } else if (bin === edges.length) {
        wideShare = 1 - lowerCdf;
      } else {
        const upperCdf = normalCdfAt(edges[bin], mean, wideDeviation);
        wideShare = upperCdf - lowerCdf;
        lowerCdf = upperCdf;
      }
      const empiricalShare = empiricalBins[bin] / SIMULATION_COUNT;
      const probability = floorProbability(
        0.68 * empiricalShare + 0.32 * Math.max(0, wideShare),
      );
      if (probability > 0) {
        sizes.push({ place: place + 1, bin, p: probability });
      }
    }
  }

  return { probabilities, top5, sizes };
}

function main() {
  const inputDirectory = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDirectory = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  const task = JSON.parse(
    fs.readFileSync(path.join(inputDirectory, "task.json"), "utf8"),
  );
  const historyText = fs.readFileSync(
    path.join(inputDirectory, "history.ndjson"),
    "utf8",
  );
  const marks = readMarks(historyText);
  if (marks.length === 0) throw new Error("History has no successful marks.");
  if (!task || !task.size || !Array.isArray(task.size.edges)) {
    throw new Error("Task is missing size edges.");
  }
  const places = Number(task.places) || PLACES;
  const candidates = collectCoins(marks);
  const result = simulate(
    candidates,
    horizonHours(task),
    task.size.edges,
    places,
  );
  const outputPath = path.resolve(
    outputDirectory,
    task.output || "forecast.json",
  );
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(result) + "\n");
}

try {
  main();
} catch (error) {
  console.error(error && error.stack ? error.stack : String(error));
  process.exitCode = 1;
}
