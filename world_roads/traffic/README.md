# Two routes home, one ETA: traffic equilibrium

Why does a drive home so often offer two completely different routes with almost the same ETA? This folder
holds three things:

* **A tested model** (`model.py`) of route choice:
  * Wardrop user equilibrium (UE) and system optimum (SO) on parallel routes with the BPR congestion curve;
  * a seeded day-to-day learning process;
  * user equilibrium on a small network with linear or BPR links (Braess's paradox, including the video's
    mythical tunnel);
  * the selection effect of an app that lists only the fastest few routes.
* **An interactive lab** (`lab/`, Vite + React + TypeScript) with four tabs. It is an optional companion
  that the video does not link to, and builds to `docs/traffic/`.
* **A captioned video** (7:40, 1920×1080), with a narration script for an optional voiceover. It opens on
  *Ever wonder why your maps app always offers two or more completely different routes home, with pretty much
  the exact same time?*, shows a real Apple Maps screenshot of exactly that (Times Square to Bayside: 59 min,
  1 hr, 1 hr 1 min), and answers *You're not alone, and your house isn't magically placed at some sweet spot in
  the traffic.* It is set on real New York roads, the same evening drive home, in the look of the light-speed
  film. Master, thumbnails and YouTube copy are in `../media/traffic/`. The same pipeline runs on a Calgary trip
  (`TRAFFIC_CITY=calgary`).

What is real and what is a toy model: [ACCURACY.md](ACCURACY.md). In short, the road geometry and the
free-flow times are real; capacities, demand and every traffic number are illustrative.

## The model

Units are minutes, and vehicles per hour over one steady peak hour.

| Function | What it does |
|---|---|
| `bpr(t0, cap, f)` | T = t0 (1 + α (f/cap)^β), with α = 0.15 and β = 4 by default |
| `parallel_ue(t0s, caps, D)` | Exact UE: find the common time λ with Σ fᵢ(λ) = D by bisection, where fᵢ(λ) = capᵢ ((λ/t0ᵢ − 1)/α)^(1/β). Used routes share λ; unused routes are no faster. |
| `parallel_so(...)` | Exact SO: the same with the marginal social cost t0 (1 + α(1+β)(f/cap)^β). Reported times are BPR times, not marginal costs. |
| `price_of_anarchy(ue, so)` | Total time at UE ÷ total time at SO (NaN without demand) |
| `joins_at(...)`, `presets(...)` | The demand at which each route starts to carry traffic, and the lab's Low / Moderate / Near-converged / Heavy presets, computed rather than hard-coded |
| `learning(...)` | Day-to-day route choice. Each day a share `replan` reconsiders, using times from `delay` days ago. Optional extras: a switching threshold `theta` (min), logit perception spread `sigma` (min), and seeded day-to-day noise. Vehicles are conserved. |
| `network_ue(net)` | UE over every simple path by pairwise path equilibration. Links are linear `(u, v, a, b)` tuples or BPR dicts `{from, to, t0, cap, plus}` (`cap=None`: fixed time). `braess_network(...)` builds the classic example; `scenario.myth_network` builds the mythical tunnel. |
| `app_shown(times, k, max_detour)` | The routes a stylized app would list |

The lab's `lab/src/model/*.ts` mirrors these line for line. `lab/src/model/model.test.ts` checks the
acceptance cases and compares against Python fixtures (`lab/scripts/fixtures.py`). The PRNG
(mulberry32) is the same in both, so a seed gives the same run in the lab and in Python.

## Run

```bash
# model tests (36)
python -m pytest -q test_model.py

# the lab
cd lab
npm install
npm run dev          # http://localhost:5173
npm test             # 26 tests, including parity with model.py
npm run build        # -> ../../../docs/traffic (static, works from any folder)

# the video
python city_routes.py fetch && python city_routes.py build         # Overture roads -> graph -> routes (TRAFFIC_CITY=nyc|calgary)
python scenario.py                                                 # every number in the video
python video.py fill 20 41                                         # stills of one section at 20 s and 41 s
python assemble.py render                                          # all sections (4 at a time) + score
python thumbs.py          # A, B, C; D wraps the maps-app screenshot (media/traffic/maps_screenshot.png)
python voiceover.py       # the narration script -> ../media/traffic/VOICEOVER.md; recorded clips in ../media/traffic/voice/ are mixed in by assemble.py
python lily.py            # records the narration with ElevenLabs' Lily voice (needs ELEVENLABS_API_KEY) -> ../media/traffic/voice/
```

The full render takes about 25 minutes on 4 cores.

## The lab

| Tab | What you can do |
|---|---|
| **A · Three competing routes** | Set demand, each route's free-flow time and capacity, and α, β. See the UE split, which routes are used, the most anyone could save by switching (0 at UE), UE vs SO flows and times, total vehicle-minutes and the price of anarchy. A demand curve marks where each route joins. Presets: Textbook (12/17/22 min), New York: JFK → Times Square and Calgary cross-town (the real free-flow times), Identical routes. |
| **B · Watch drivers find equilibrium** | Start, pause, step, reset and speed controls. Set the share who re-plan, reaction delay, switching threshold, perception spread, noise and seed. Presets cover "Settles", "Keeps wobbling", "Everyone chases yesterday's winner", "Ignore under 2 min", "Fuzzy perceptions + noise" and "Stale information". The exact UE is drawn as a dashed target. |
| **C · Braess's paradox** | A big connector toggle and the number of drivers; advanced link costs. A path table shows used and unused paths. A chart of average trip vs demand shades the band where the connector hurts (3,000–9,000 drivers with the classic numbers). |
| **D · What the app shows** | Many candidate routes with no traffic at all (random, or 400 real routes at 3 a.m. in New York or Calgary). Choose how many the app lists, a detour limit and rounding. Compares the spread of the listed ETAs with the spread of all routes, over 1,000 seeded random cities. |

Settings are saved in the browser's localStorage only. *Reset everything* clears them.

**Questions to try.** Each is answerable without touching code:
1. Raise demand from 0 (tab A). B and C join at the marked points, and afterwards all used routes share
   one time.
2. Move route C's free-flow time and watch its joining demand move.
3. Raise A's capacity: both the split and the average change.
4. Watch the price-of-anarchy tile across demand. It is largest in moderate traffic.
5. In tab B, "Everyone chases yesterday's winner" versus "Settles".
6. Set a 1–2 min switching threshold: the routes freeze with unequal times.
7. In tab C, toggle the connector at 2,000, 4,000 and 10,000 drivers.
8. In tab D, try 3 listed out of 20, then 3 out of 100.

## Changing the scenario

* **Lab defaults:**
  * shared route sets and demand: `lab/src/scenarios.ts`;
  * each tab's defaults: the `DEFAULT_*` constant at the top of `lab/src/modes/Mode*.tsx`.
* **The video's traffic** (capacities, rush-hour demand, the learning runs): constants at the top of
  `scenario.py`. The mythical tunnel (which bottlenecks it joins, its minutes, their capacities and the
  demand) is the `myth` entry in `cities.py`. Re-run `scenario.py`, then `assemble.py render`.
* **The trip and the city:** one entry per city in `cities.py`:
  * the fetch box, projection centre and the trip's two ends;
  * default speeds;
  * rush-hour demand and the settling learning run;
  * the words the video uses for the three routes.

  `TRAFFIC_CITY` picks the city (default `nyc`). How `build` chooses the three routes depends on the
  entry. With `crossings`, it takes the fastest route through each named bridge or tunnel (New York).
  Otherwise it takes the three least-overlapping routes within 15% of the fastest (Calgary).
* **Another city:** add an entry to `cities.py`, then run `city_routes.py fetch` and `build`,
  `scenario.py` and `assemble.py render`. The map views fit the new routes automatically.

## Next directions (not built yet)

* Shared bottlenecks: a small editable network with Frank-Wolfe or path-based UE, so route times are
  correlated.
* Demand elasticity and induced travel; tolls at marginal cost to push UE towards SO.
* Two origin–destination groups sharing a road.
* Information latency: many drivers receiving the same suggestion at once.
