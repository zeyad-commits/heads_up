# SDPLO-RangeEV — Heads-Up Short Deck PLO5 bot

A range-aware expected-value bot for the TAMU Datathon Poker Bot Challenge.
`agent.py` keeps the starter's server plumbing untouched; only the three
strategy functions (and the team/bot names) were filled in. They delegate to
the `sdplo/` package.

> **Before submitting:** set `PARTICIPANT` in `agent.py` to your registered
> team name, then run `python -m pytest -q` and `./tools/check_build.sh`.

## Quick start

```bash
pip install -r requirements-dev.txt          # fastapi, uvicorn, pytest, httpx (numpy is in the base image)

python -m pytest -q                          # official server tests (test_agent.py)
python -m pytest -q tests/                   # strategy tests (21 tests)

python agent.py                              # bot on :5008
python baseline-bots/baseline_random.py      # opponent on :5009
python judge_engine.py --deals 1000          # full duplicate match
```

`requirements.txt` adds only `numpy` (unpinned, so the build reuses the copy
already in the tournament base image).

## Architecture

```
judge ──HTTP──> agent.py (unchanged plumbing)
                   │  new_half / choose_action / on_hand_end
                   ▼
              sdplo/bot.py  BotState ─── time governor + try/except + legality guard
                   │
   ┌───────────────┼───────────────────────────┬──────────────────────────┐
   ▼               ▼                           ▼                          ▼
history.py     opponent.py                ranges.py                  policy.py
action log →   online frequencies         particle filter over       EV of each legal action
contexts       (fold/call/raise per       opponent hands + Monte     (pot odds, fold equity,
               context, bluff share)      Carlo equity               realised equity) → choice
                                               │
                                     fasteval.py + tables.py
                                     vectorised evaluator, preflop tables
```

Game-state processing (`history.py`), opponent statistics (`opponent.py`),
range/equity estimation (`ranges.py`) and the decision rule (`policy.py`) are
separate modules; `bot.py` is the only glue, and `agent.py` the only interface.

### 1. Hand evaluation (`fasteval.py`)
A five-card score depends only on the rank multiset and whether all five
cards share a suit. Both are encoded additively (`Σ5^rank`, `Σ8^suit`), so a
hole pair's partial key plus a board triple's partial key gives the full key
by plain addition. numpy broadcasts the 10 × 10 Omaha (exactly 2 + 3)
combinations for whole batches of hands in one gather. Scores come from the
official `evaluator._score_ranks` and are tested equal to `omaha_score`;
batch speed is ~1 µs/hand vs ~38 µs for the reference.

### 2. Preflop tables (`tables.py`, `training/build_preflop.py`)
All 376,992 five-card hands (19,998 suit classes) have precomputed equity vs a
random hand and vs a top-25% hand (3,000 Monte-Carlo runouts each). Their
blend's percentile is "preflop strength". 1 MB, loaded at startup.

### 3. Opponent range: a Bayesian particle filter (`ranges.py`)
Each hand, ~600 candidate opponent hands are sampled from the unseen cards
(never our cards or the board — no information leakage). Each opponent action
multiplies a particle's weight by P(action | hand strength):

* the opponent's observed fold / call / raise frequencies in that betting
  context (`opponent.py`) define strength *bands*: bluffs at the bottom,
  folds, calls, value raises at the top, with soft logistic edges;
* strength is the particle's percentile **within the opponent's current
  weighted range** on that street (preflop table; made hand + draw outs on
  flop/turn; exact score on the river), so each action narrows the range
  relative to what survived the previous one.

Because the bands come from learned frequencies, the filter adapts on its
own: a calling station's calls carry no information, a nit's bet means a
strong range, a maniac's raises are discounted by his measured bluff share.

### 4. Decision: expected value per action (`policy.py`)
Equity vs the weighted range comes from Monte-Carlo runouts per particle.
For each legal action:

| action | EV from now (sunk chips ignored) |
|---|---|
| fold | 0 |
| check | `E_r · pot` |
| call | `E_r · (pot + call) − call` (exact all-in / short-stack handling) |
| raise | `P(fold)·pot + (1−P(fold))·(E_c,r · pot_called − chips_added) − reraise_cost` |

`P(fold)` and `E_c` (equity vs the hands that continue) are computed from the
same particles, re-weighted by the opponent's continue-likelihood at that
bet size. Value bets, bluffs and semi-bluffs therefore all come from one
formula: we bluff exactly when fold equity pays for it (never vs a calling
station), value-bet thinner vs stations, and fold more to nits.
`E_r` scales equity by a tuned realisation factor per street/position when
cards remain (1.0 on the river or when all-in). A small softmax temperature
mixes near-equal choices so the bot is not perfectly predictable.

### 5. Robustness and time
* everything runs in `try/except`; failures fall back to a cheap rule bot,
  and the answer is always checked against `obs.legal_actions`;
* a **time governor** tracks thinking time: above a pace of 4 s / 1000 hands
  it switches to fewer particles; above 6.5 s it plays the cheap rule bot,
  so the 8 s bank never runs out;
* numpy first-call costs are paid during startup by a warm-up.

## Training (`training/`)
| script | what it does |
|---|---|
| `build_preflop.py` | builds `sdplo/data/preflop.npz` (≈3 min on 2 cores) |
| `sim.py` | in-process duplicate matches on the official `game.py` (same rules, seat swap, errors, verdict) |
| `opponents.py` | training pool (random, passive, aggressive, tight, equity-based) and held-out pool |
| `tune.py` | cross-entropy-method search over strategy parameters with self-play league |
| `evaluate.py` | tournament-style evaluation on separate `eval-` seeds incl. held-out opponents |
| `timing.py` | thinking time per decision / per 1000 hands |

```bash
python training/build_preflop.py --samples 3000 --seed 1
python training/tune.py --generations 8 --pop 12 --deals 300 --seed 0   # writes checkpoints/gen_XX.json
cp checkpoints/gen_07.json sdplo/params.json                            # adopt a checkpoint
python training/evaluate.py --deals 1000 --seeds 2 --vs checkpoints/v0_defaults.json
```

Tuning fitness = Σ over opponents of `3·tanh(z/3)` with `z = mean/SE` of
bb/deal — it rewards turning chops into statistically significant wins
(how the tournament scores) and saturates on opponents already beaten, so the
search cannot trade a loss to a strong bot for a bigger win over a weak one.
All candidates in a generation play the same seeds (common random numbers);
held-out bots and `eval-` seeds are never used in tuning.

## Why this architecture (and not CFR / Deep CFR / NFSP)

| approach | verdict for this event |
|---|---|
| Vanilla CFR | Infeasible: 376,992 hole hands × boards; needs heavy card abstraction. |
| MCCFR + card abstraction | Betting tree is small (4 actions, raise cap 4, 40 bb), but a useful blueprint needs millions of iterations; this was built on 2 CPU cores in pure Python/numpy. Coarse buckets lose exactly the draw/nut information PLO turns on, and an equilibrium blueprint does not *exploit* the weak bots that decide most hackathon matches. |
| Deep CFR / NFSP / self-play RL | Large CPU training budgets, unstable without careful tuning; inference fits the time bank, but there is little reason to expect it to beat an exact-equity model in a hackathon timeframe. |
| **Range-aware EV (chosen)** | Exact hand evaluation and Monte-Carlo equity, Bayesian opponent-range narrowing, explicit pot odds and fold equity, online opponent adaptation, and a handful of parameters tuned by self-play/population matches. Cheap enough for the 8 s bank, adapts to any opponent within a few dozen hands. |

The scoring rule shapes this: a match is won only with a **statistically
significant** edge, and every bot plays every bot once. Reliably beating the
many simple bots (and chopping, not losing, against strong ones) is worth more
than marginal equilibrium accuracy.

## Results (measured, not projected)

All matches are full duplicate matches (1,000 deals × 2 halves, seats
swapped), scored like the tournament. In-process results use `training/sim.py`
on `eval-` seeds that tuning never saw. "Held-out" bots were never used in
tuning.

**Tuned parameters (shipped, `sdplo/params.json`), 2 seeds each.** Nit, maniac,
equity bot, TAG and v0 rows are from the final config
(`training/results/eval_final.txt`); the rest from the same parameters before
the time-governor cap was lowered (`eval_gen07.txt`):

| opponent | bb/deal (seed 0 / seed 1) | verdict |
|---|---|---|
| starter (check/fold) | +1.50 / +1.50 | WIN / WIN |
| guide example | +3.28 / +3.30 | WIN / WIN |
| random | +10.60 / +7.50 | WIN / WIN |
| calling station | +10.92 / +10.84 | WIN / WIN |
| always half-pot | +13.30 / +13.32 | WIN / WIN |
| always pot | +7.42 / +8.97 | WIN / WIN |
| nit | +2.50 / +3.25 | WIN / WIN |
| maniac | +10.30 / +9.16 | WIN / WIN |
| equity bot (pot-odds vs random hand) | +2.85 / +3.13 | WIN / WIN |
| **TAG (held-out)** | +1.44 / +1.65 | WIN / WIN |
| **sticky (held-out)** | +11.01 / +9.95 | WIN / WIN |
| untuned v0 (previous version) | +0.20 / −0.13 | CHOP / CHOP |

Untuned v0 on the same seeds scored 21.5 / 22 (chop vs TAG on seed 0,
+0.81 bb/deal). Tuning's gain is real but small: better results vs the
held-out TAG bot, and a statistical tie head-to-head with v0.

**Over real HTTP with the official `judge_engine.py`** (halves played
separately with fresh servers, seed `http7`):

| opponent | bb/deal | 95% CI | result | judge-charged thinking time per half |
|---|---|---|---|---|
| baseline_random | +9.62 | [+8.09, +11.16] | AGENT1_WIN | 3.13 s / 2.97 s |
| baseline_always_pot | +7.26 | [+5.67, +8.84] | AGENT1_WIN | 2.91 s / 2.61 s |
| baseline_calling_station | +10.53 | [+9.53, +11.52] | AGENT1_WIN | 3.15 s / 3.10 s |
| baseline_always_half | +10.23 | [+8.91, +11.55] | AGENT1_WIN | 3.26 s / 3.37 s |

0 errors in every half. Under deliberate CPU contention (two busy processes on
the 2-core test machine) the governor kept the worst half at 5.88 s of 8 s.

Tests: `test_agent.py` 12/12, `tests/test_sdplo.py` 21/21.

**Caveat on the HTTP numbers:** PyPI was unreachable in the environment where
this was built, so the HTTP runs used a minimal stand-in for FastAPI/uvicorn.
The strategy code and the judge were the real ones, and the official tests
passed through the stand-in, but **run `python -m pytest -q` and
`./tools/check_build.sh` with the real packages before submitting.** Real
uvicorn is typically faster than the stand-in, so time use should be the same
or lower.

## Key strategic decisions
* **Exploit through measured frequencies, not hard-coded styles.** The same
  code bluffs a folder, never bluffs a station, and folds to a nit's bets,
  because fold equity and range narrowing both come from what this opponent
  has actually done this half.
* **Equity vs the range that continues.** Value bets are sized by equity
  against hands that call, not against all hands.
* **Realised equity, not raw equity, when cards remain.** Out of position and
  preflop, hands realise less than their all-in equity; the factors were tuned.
* **Robustness first.** Three fallback layers, a time governor, warm-up at
  startup, and an always-legal guard: a forfeit costs the whole match.

## Known limitations and future work
* **One-step lookahead.** EVs assume the hand is decided after the current
  bet (re-raises only enter as a penalty). A small river/turn subgame solver
  would improve multi-street play, especially vs strong bots.
* **Head-to-head vs strong bots is unknown.** Against a copy of itself it
  chops, as expected; there are no other strong opponents to test against. An
  adaptive opponent could in principle exploit its fairly deterministic play;
  the softmax temperature is small.
* **Draw strength is approximate.** Outs are counted by rank/suit heuristics
  (no nut-draw distinction, no blockers).
* **Opponent model resets every half** (tournament rule) and starts from
  priors; the first ~50 hands are played closer to the prior model.
* **Tuning is noisy.** 8 CEM generations × 300-deal matches on 2 cores;
  longer runs with more deals and more league opponents would tighten it.
* **No exploitability measurement.** A true best-response computation is not
  feasible for this game here; robustness is approximated by the varied
  opponent pool, held-out bots and self-play.
