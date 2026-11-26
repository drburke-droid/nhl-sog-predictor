# Results: did the agents invent hockey positions?

Main run `runs/main`: 27.7M decisions of self-play (15,377 three-minute games,
about 769 hours of game time), 1,802 PPO iterations on a 4-core CPU in about
two hours. Evaluation: 128 games per policy (`analyze.py`). Interactive page:
`viewer/` (index.html + data.json).

## Short answer

**Yes, at first. Then they gave them up.**

* **~3M decisions (peak):** the five skater slots sorted themselves into a
  complete line-up.

  | slot | most often | share of 5-on-5 frames |
  |---|---|---|
  | 1 | left wing | 70% |
  | 2 | centre | 51% |
  | 3 | left defence | 74% (defence 97%; keeps that label ~2 min at a time) |
  | 4 | right defence | 47% (defence 67%) |
  | 5 | right wing | 53% |

  Role consistency 59% in evaluation (67% smoothed during training). On this
  measure 20% means interchangeable players and 100% means rigid positions.
  The team attacked three back, two up, and played safe, slow hockey: 0.6
  goals, 4 shots and 2 passes per team per game, and 2.2 offsides.

* **27.7M decisions (final):** role consistency 28%, defence/forward
  consistency 63% (52% = none). Every skater rotates through every spot, and
  the deepest skater changes every 7 to 10 seconds. With the puck the team
  keeps a staggered "ladder" about 10 ft apart: +4, −17, −27, −36 and −48 ft
  from the puck. There is no 2-and-3 split. Slot 2 is the only identity left:
  it is the highest skater 45% of the time, a stretch forward. Without the
  puck nobody retreats; on average all five are up-ice of it.

* **The positionless team is much better.** Head-to-head over 128 games, final
  policy vs the peak-positions policy: **120 wins, 0 losses, 8 ties**, average
  score 3.0–0.2 (`runs/main/head_to_head.json`). It plays more open hockey:
  1.8 goals, 9.7 shots and 11.5 passes (83% completed) per team per game,
  0.5 offsides, and a 0.904 save percentage.

## The goalie pull

The coach (rewarded only for winning) learned **never** to pull the goalie,
even down a goal with seconds left. Forcing the pull with the trained skaters
(`coach_check.py`) shows it was right for this team:

| | goals for / min | goals against / min |
|---|---|---|
| 5-on-5 | 0.60 | 0.60 |
| goalie pulled (6-on-5) | 0.51 | 1.99 |

The extra attacker *lowers* scoring. Down one goal, pulling cuts the chance
to tie at every time left: with 30 s left it is 14% pulled vs 26% at 5-on-5.
This is a chicken-and-egg trap. Slot 6 (the extra attacker) almost never got
ice time, so the skaters never learned 6-on-5, so pulling never paid.

## Why the positions may have faded

These are hypotheses, each testable by adding the missing ingredient:

* **Identical players.** Real positions rest on differences: shooting hand,
  size, skating, faceoffs. Here every agent is the same clone, so a fixed role
  only buys coordination.
* **One shared network** can coordinate by situation instead of by name
  (nearest skater takes the puck, the rest fill the ladder).
* **No fatigue, line changes or penalties**, which in real hockey make
  holding a lane cheap and getting caught up-ice costly.
* **Self-play against a mirror**: one style meeting itself, not a league of
  different opponents.

Next experiments: left/right handedness per slot, heavier slower bodies for
some slots, fatigue and line changes, population-based opponents, and
deliberate 6-on-5 practice.

## Development notes (the exploits the agents found)

The rules changed during development as the agents exposed simulator
artifacts. Each fix is in the git history.

1. Free-roaming goalies walked to centre ice and kicked the puck back and
   forth; skaters guarded empty nets. → goalies confined to their net area.
2. Learned goalies stayed near-random (~35% saves), shot quality stopped
   mattering, and play became a swarm; the coach then pulled constantly
   because the goalie was worth nothing. → scripted angle-cutting goalie
   calibrated to ~90% on set shots and ~50% goals on cross-ice one-timers.
3. Passes to moving teammates almost never connected (bad lead calculation,
   too much aim noise), and every stick near a lane was a certain interception
   even at the moment of release. Passing died out. → intercept-point lead,
   2° noise, receiver reach, probabilistic pick-offs with a 0.1 s reaction
   window.
4. A puck-x progress bonus taught carriers to park the puck behind the
   opponent's net and never shoot. → progress measured as distance to the goal
   mouth, tiny possession bonus.

The bootstrap reward (team-level only, never naming a player) was on for the
first 4M decisions and faded to zero by 12M. The peak line-up formed while it
was on, and the positionless style developed after it was gone.

## Files

* `runs/main/final_policy.pt`, `runs/main/peak_policy.pt`: network weights
  (load with `analyze.load_policy`).
* `runs/main/log.jsonl`: per-iteration training log, including role counts.
* `runs/main/coach_check.json`, `runs/main/head_to_head.json`: the two tests
  above.
* `viewer/data.json`: everything the results page shows.
