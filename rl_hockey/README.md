# Hockey self-play lab: do learning agents invent positions?

A multi-agent reinforcement-learning experiment. Two teams of agents (5 skaters
+ 1 goalie each) play hockey on a simulated top-down rink. Only the rules are
hard-coded. Nothing about positions, formations or roles is coded anywhere, and
the question is whether the agents invent them: a pair of defencemen, a centre
and two wingers, or something else.

## What is hard-coded

`hockey_env.py`: a vectorised 2-D simulator (many games in lock-step with numpy).

* **Rink**: 200 x 85 ft, 28 ft corner radius, goal lines 11 ft from the boards,
  blue lines 25 ft from centre, 6 ft nets, creases, faceoff dots.
* **Physics**: players are discs with momentum (acceleration, top speed, drag,
  board bounces, body contact). The puck slides with low friction, bounces off
  the boards and the net frame, and is carried a stick's length in front of its
  carrier, who skates slightly slower.
* **Puck play**: a loose puck is collected by the first stick it reaches. Hard
  pucks are harder to control, and bodies block shots. Passes travel at
  60 ft/s with ~2° of aiming noise toward the receiver's interception point.
  The intended receiver reaches 4 ft for the puck. Anyone else whose stick
  (2.5 ft) is in the lane picks a moving puck off only sometimes: about 85%
  when the lane runs through his stick, 25% at the edge of his reach, and
  never in the first 0.1 s of flight, before he could react. A slow loose puck
  is always collected. (An earlier version made every stick near the lane a
  certain interception; passing died out and play turned into a swarm.) Opponents within 3 ft
  can strip the carrier (about 55% a second for one checker), and a hard hit
  can knock the puck loose. A goalie
  saves 90% of hard pucks he gets a piece of; near his crease he freezes half
  of them, otherwise he gives up a rebound.
* **Rules**: goals; immediate offside; no-touch icing; goalie freezes; faceoffs
  with every player on his own side of the dot. Goalies stay within 12 ft of
  the middle of their own goal line, in front of it, a stand-in for the crease
  and trapezoid rules. Free-roaming goalies first walked to centre ice to kick
  the puck around, then drifted out of position, which made pulling the
  goalie almost free.
* **Goalie pull**: a team's coach can pull the goalie at any moment. An extra
  attacker (skater slot 6) comes off the bench, and the goalie can only return
  at a stoppage.
* **No imposed line-up**: after every goal, each skater is re-spawned at a
  random spot in his own half. Faceoffs only require players to be on their
  own side of the dot.

## The agents

`train.py`: self-play PPO. Both teams share the same three policies.

| Policy | Acts | Sees | Rewarded for |
|---|---|---|---|
| skater (one network for all 6 skater slots) | every 0.1 s: skate direction (8) / glide / brake; with the puck: keep, shoot, dump left, dump right, pass to slot *j* | the whole game state in its own team's frame (always attacking +x), plus a one-hot **slot id** | +1 goal for, −1 goal against |
| goalie | every 0.1 s: skate direction / glide / brake | same | +1 / −1 per goal |
| coach | every 1 s: goalie in / out (starts learning after 12M steps, once the players can play) | score, time left, possession, puck position | **winning**: +1 win, 0 tie, −1 loss |

The slot id is the only thing that tells two skaters apart. Slots *can*
specialise, but nothing tells them how to. Each team sees the game rotated
180 degrees, so "left" means the same thing to both teams, and the bench side
is randomised per game.

**Bootstrap reward.** Random agents almost never touch the puck, so early on a
small extra reward helps learning get started. It has three parts: a bonus for
having possession, a bonus for moving the puck toward the other net, and a
bonus when the team's closest skater gets nearer a loose puck. All three are
team-level and say nothing about *which* player should do anything. The bonus
is at full weight for the first 4M game steps, fades to zero by 12M, and from
then on goals are the only reward.

## How positions are measured

Every frame of 5-on-5 live play, the five skaters are labelled by where they
stand relative to each other. The two deepest are **D**: the one with larger y
(further left when attacking) is LD, the other RD. The other three are
**forwards**, ordered left to right as LW, C, RW. This gives a slot × label
count table.

* **Role consistency**: the share of frames in which each slot holds its own
  label, under the best one-to-one slot→label assignment. Positionless play
  (interchangeable slots) scores 20%; rigid positions score 100%.
* **D/F consistency**: the same, but only asking whether the same two slots
  are the deepest. Positionless scores 52%.

## Running it

```bash
pip install torch numpy                      # CPU is fine
python train.py --run runs/main --hours 2 --lr 5e-4 --ent 0.005   # resumes from runs/main/latest.pt
python analyze.py --run runs/main            # writes runs/main/analysis.json
python viewer/build_data.py runs/main/analysis.json viewer/data.json
# then serve viewer/ (index.html + data.json) with any static web server
```

Training throughput on a 4-core CPU is about 3.5k game steps per second
(≈13M per hour).

## Results

See `RESULTS.md` (written once the main run finished).
