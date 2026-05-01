# Chess experts findings

## Iteration log

### Iteration 1 - chapter split

Created an expert-only chapter. This deliberately avoids Phi and any in-house
language model. The working question is:

```text
Can we build better chess organs first, then later decide how to attach them?
```

The initial three subsections are:

```text
1. teacher distillation from a chess engine
2. JEPA-style latent board dynamics
3. broader mate-in-one motif generalization
```

### Iteration 2 - teacher distillation adapter

Implemented:

```bash
.venv/bin/python experiments/12_chess_experts/chess_teacher_distill.py
```

The script is ready to distill from a UCI chess engine:

```text
random legal FEN positions
-> Stockfish or another UCI teacher chooses a move
-> board-feature MLP imitates the teacher move
-> held-out positions are scored both by raw argmax and legal-masked move
```

Current local result:

```text
status: blocked
reason: UCI engine not found
```

This is the right failure mode. We do not have a Stockfish binary in this
environment, so the chapter records that dependency explicitly instead of
pretending that python-chess alone is a chess teacher.

### Iteration 3 - JEPA board-transition bridge

Implemented:

```bash
.venv/bin/python experiments/12_chess_experts/chess_jepa_bridge.py
```

This trains an expert-only JEPA-style bridge:

```text
current board features -> encoder -> z_t
legal move features + z_t -> predictor -> predicted z_{t+1}
next board features -> encoder -> target z_{t+1}
```

Result:

```text
train pairs: 2,880
held-out pairs: 720
runtime: 4.42s
held-out cosine: 0.9986
normalized MSE: 0.000042
nearest-neighbor next-board retrieval: top1=0.667, top5=0.997
```

Interpretation:

This is a useful non-language bridge. It shows that a small latent transition
model can predict the next board representation from current board plus move.
The top-5 retrieval number is especially promising. The top-1 misses are not
catastrophic yet; many chess states share very close material/geometry, so the
next step should include harder negative sampling rather than just MSE/cosine.

### Iteration 4 - broader mate-in-one motif generalization

Implemented:

```bash
.venv/bin/python experiments/12_chess_experts/chess_motif_generalization.py
```

This broadens the previous mate-in-one expert from one back-rank motif to four
families:

```text
back_rank_rook_queen
queen_side_file
rook_side_file
knight_corner
```

Result:

```text
model params: 7,887,616
training cases: 2,048
held-out cases: 128
runtime: 5.70s
exact move pass: 126/128 = 0.9844
legal mate pass: 128/128 = 1.0000
```

Per-family legal mate pass:

```text
back_rank_rook_queen: 32/32
queen_side_file:      32/32
rook_side_file:       32/32
knight_corner:        32/32
```

Interpretation:

This is stronger than the Chapter 11 chess expert. It is still a generated
mate-in-one curriculum, but it now covers multiple tactical motifs and reports
per-family performance. The two exact misses are acceptable because the
positions had multiple mating moves; the predicted moves were still legal
checkmates.

### Iteration 5 - diagnostic arena, not yet adversarial play

Implemented:

```bash
.venv/bin/python experiments/12_chess_experts/chess_expert_arena.py
```

Result:

```text
status: diagnostic_arena_not_full_game
positions: 64
motif policy legal mate pass: 63/64 = 0.9844
JEPA transition cosine on motif moves: 0.9774
JEPA transition normalized MSE on motif moves: 0.00070485
```

Interpretation:

The motif expert and JEPA expert cannot honestly "play against each other" yet.
The motif expert is a move policy for mate-in-one. JEPA is a world model: it
predicts the next latent board state for a given move, but it does not choose
winning moves by itself.

A true adversarial benchmark needs:

```text
expert(board) -> legal move
alternate turns
terminal result: win/loss/draw
many games from varied starts
```

The next step is to add a value or policy head on top of JEPA so it can choose
moves. Then the adversarial arena can be real: motif-policy or distilled-policy
versus JEPA-policy, scored by game result.

### Iteration 6 - JEPA-backed policy arena

Implemented:

```bash
.venv/bin/python experiments/12_chess_experts/chess_policy_arena.py --freeze-encoder --sample-rows 8
```

This adds the missing `expert(board) -> legal move` interface for JEPA:

```text
JEPA board encoder -> frozen latent -> policy head -> legal-masked move
```

The motif expert and JEPA-policy expert are then evaluated on the same held-out
mate-in-one positions. This is still not full opening-to-endgame chess, but it
is now a competitive tactical policy arena: each expert chooses a move, and the
arena scores whether that move immediately checkmates.

Result:

```text
positions: 96
motif wins: 1
JEPA-policy wins: 0
ties, both mate: 95
ties, both fail: 0
motif mate rate: 1.0000
JEPA-policy mate rate: 0.9896
```

Per-family:

```text
back_rank_rook_queen: motif 1.0000, JEPA-policy 0.9583
knight_corner:        motif 1.0000, JEPA-policy 1.0000
rook_side_file:       motif 1.0000, JEPA-policy 1.0000
queen_side_file:      motif 1.0000, JEPA-policy 1.0000
```

Interpretation:

The motif policy still wins narrowly on this tactical arena, but the frozen
JEPA encoder plus a small policy head is already competitive. The key upgrade
is architectural: JEPA is no longer only a transition model. It now has a move
selection interface and can participate in adversarial policy benchmarks.

The next step toward a true game benchmark is to train a value head or
multi-ply policy and play many short games from tactical starts, scoring
terminal win/loss/draw instead of only mate-in-one success.

### Iteration 7 - multi-ply puzzle sequence arena

Implemented:

```bash
.venv/bin/python experiments/12_chess_experts/chess_puzzle_sequence_arena.py --freeze-encoder
```

This moves beyond one-move puzzles. The arena constructs positions with a
defender reply and a final mating move:

```text
black legal reply
white expert move
terminal checkmate required
```

Both experts receive the same puzzle position and must complete the line.

Result:

```text
puzzles: 48
motif solved: 47/48 = 0.9792
JEPA-policy solved: 46/48 = 0.9583
motif wins: 1
JEPA wins: 0
ties both solve: 46
ties both fail: 1
```

Example solved line:

```text
FEN: 8/5pp1/8/6pk/6p1/8/8/K2R4 b - - 0 1
target line: g7g6, d1h1
motif: g7g6, d1h1 -> solved
JEPA:  g7g6, d1h1 -> solved
```

Interpretation:

This is now a puzzle benchmark rather than a single-move classifier. It is
still constructed and narrow, but it is closer to adversarial evaluation:
there is an opponent move, a sequence to complete, and terminal checkmate is
the success criterion. The motif expert remains slightly ahead, while the
JEPA-policy remains competitive.

### Iteration 8 - sample-efficiency competition sweep

Implemented:

```bash
.venv/bin/python experiments/12_chess_experts/chess_competition_sweep.py --budgets 4,8,16,32,64,128 --policy-epochs 80
```

This makes the two chess experts compete across increasing data budgets. The
test set is fixed at 192 held-out mate-in-one tactical positions. At each
budget, both experts train on the same number of examples per motif:

```text
motif-policy: direct board -> move classifier
JEPA-policy:  frozen JEPA board encoder -> policy head -> move classifier
```

Result:

```text
train per motif | motif mate rate | JEPA-policy mate rate | winner
4               | 0.2083          | 0.2083                | tie
8               | 0.2812          | 0.2865                | JEPA-policy, slight
16              | 0.4844          | 0.4688                | motif-policy, slight
32              | 0.7031          | 0.7031                | tie
64              | 0.9115          | 0.8958                | motif-policy, slight
128             | 0.9635          | 0.9740                | JEPA-policy, slight
```

At the largest budget:

```text
positions: 192
motif wins: 2
JEPA-policy wins: 4
ties both mate: 183
ties both fail: 3
```

Interpretation:

There is no dramatic emergent chess intelligence yet. There is, however, a
real competitive signal: the frozen JEPA encoder is not just reconstructing
board transitions anymore. With a small policy head, it becomes a tactical
move selector and eventually edges the direct motif classifier on this held-out
suite.

The honest conclusion is budget-dependent:

```text
multi-ply sequence arena: motif-policy is ahead, 47/48 vs 46/48
sample sweep at 128 examples per motif: JEPA-policy is ahead, 0.9740 vs 0.9635
```

This is still a generated tactical domain. The next harder version should make
the competition more adversarial by using longer forced lines, mixed tactical
families, and failure cases mined from the current decisive examples.

### Iteration 9 - harder multi-seed competition sweep

Implemented a stronger sweep:

```bash
.venv/bin/python experiments/12_chess_experts/chess_competition_sweep.py \
  --seeds 67,68,69 \
  --budgets 16,32,64,128,192,256 \
  --max-train-per-family 256 \
  --val-per-family 96 \
  --policy-epochs 140 \
  --jepa-epochs 18 \
  --jepa-pairs 2400 \
  --jepa-val-pairs 400
```

This increases:

```text
seeds: 1 -> 3
held-out positions per run: 192 -> 384
aggregate held-out positions per budget: 1,152
max training examples per motif: 128 -> 256
max total policy examples: 512 -> 1,024
policy epochs: 80 -> 140
JEPA bridge epochs: 14 -> 18
JEPA bridge pairs: 1,800 -> 2,400
```

Aggregate result:

```text
train per motif | motif mean | JEPA-policy mean | delta JEPA-motif | winner
16              | 0.5148     | 0.5147           | -0.0000          | motif, by rounding
32              | 0.7188     | 0.7057           | -0.0130          | motif
64              | 0.8550     | 0.8559           | +0.0009          | JEPA-policy
128             | 0.9540     | 0.9462           | -0.0078          | motif
192             | 0.9792     | 0.9792           | +0.0000          | tie
256             | 0.9896     | 0.9835           | -0.0061          | motif
```

At 256 examples per motif:

```text
runs: 3
positions scored: 1,152
motif-only wins: 14
JEPA-only wins: 7
ties both mate: 1,126
ties both fail: 5
```

Interpretation:

The single-seed JEPA edge did not survive as a stable claim under the harder
multi-seed benchmark. The better statement is:

```text
Both experts become very strong on this generated tactical distribution.
JEPA-policy is genuinely competitive and sometimes wins a budget slice.
The direct motif policy remains slightly stronger at the highest budget.
```

That is still progress. The frozen JEPA encoder is viable as a policy substrate,
but the current arena is now close to saturated. The next benchmark should not
just add more epochs. It should increase task hardness: longer forced lines,
more distractor pieces, mixed motifs in one position, and adversarially mined
failure cases.

### Iteration 10 - full legal game arena

Implemented:

```bash
.venv/bin/python experiments/12_chess_experts/chess_full_game_arena.py \
  --games 24 \
  --max-plies 220 \
  --opening-plies 6 \
  --train-per-family 1024 \
  --max-train-per-family 1024 \
  --policy-epochs 240 \
  --jepa-epochs 30 \
  --jepa-pairs 10000 \
  --jepa-val-pairs 1000 \
  --anti-repetition 2.0
```

This is the first full-game benchmark in this chapter. It is not a full chess
engine yet. Both experts are still trained from generated tactical data, then
forced to play complete legal games. The arena adds:

```text
paired starts with color swaps
legal move masking for every move
anti-repetition pressure
checkmate detection
draw detection
material adjudication at the ply cap
SAN move logs for every game
```

Training and benchmark scale increased again:

```text
training examples per motif: 256 -> 1,024
total tactical policy examples: 1,024 -> 4,096
JEPA bridge pairs: 2,400 -> 10,000
JEPA bridge epochs: 18 -> 30
policy epochs: 140 -> 240
full games: 0 -> 24
max plies per game: 220
```

Result:

```text
games: 24
motif wins: 5
JEPA wins: 4
draws: 15
average plies: 77.29
checkmates: 6
claimable draws: 15
material adjudications: 3
```

JEPA bridge pretrain:

```text
held-out cosine: 0.998508
nearest-neighbor top1: 0.5700
nearest-neighbor top5: 0.9880
```

Example decisive game:

```text
start: rnbqkbnr/ppppp2p/6p1/5p1Q/1P2P3/8/P1PP1PPP/RNB1KBNR w KQkq - 0 4
white: motif
black: JEPA
winner: JEPA by checkmate in 20 plies
line: Nc3 gxh5 Nd1 fxe4 Be2 d5 Rb1 Qd7 Ra1 Qa4 Ba6 bxa6 Rb1 Bg4 Ra1 Nc6 h4 Nxb4 c4 Qxd1#
```

Interpretation:

This is a major benchmark upgrade because the experts now have to survive a
whole legal chess game instead of solving isolated mate positions. It also
shows the current limitation clearly: full-game behavior is still brittle and
draw-heavy. The experts know tactical move shapes, but they do not yet have a
stable opening, middlegame, endgame, or anti-loop policy.

The honest score is close:

```text
motif: 5 wins
JEPA: 4 wins
draw: 15
```

The next improvement should train on full-game state/action traces, not only
mate-in-one motifs. The arena now exists, so future experts can be judged by
game result rather than by tactical classification alone.

### Iteration 11 - full-game trace policies

Implemented:

```bash
.venv/bin/python experiments/12_chess_experts/chess_full_game_trace_arena.py \
  --teacher-games 180 \
  --max-trace-cases 9000 \
  --games 32 \
  --max-plies 240 \
  --jepa-pairs 12000 \
  --jepa-val-pairs 1200 \
  --jepa-epochs 32 \
  --policy-epochs 120 \
  --freeze-encoder
```

This directly attacks the previous limitation. Instead of training only on
mate-in-one tactical motifs, the script generates full-game state/action traces
with a shallow legal chess teacher:

```text
current board
legal move candidates
material delta
check/checkmate pressure
castling and promotion bonuses
king safety
mobility
anti-repetition penalty
selected teacher move
```

Then it trains two policies on the same trace set:

```text
direct_full_trace: board features -> move class
jepa_full_trace:   frozen JEPA board encoder -> move class
```

Trace curriculum:

```text
trace cases: 9,000
opening: 725
middlegame: 961
endgame: 7,314
```

Full-game result:

```text
games: 32
direct_full_trace wins: 7
JEPA_full_trace wins: 9
draws: 16
decisive games: 16
average plies: 71.38
checkmates: 16
claimable draws: 16
```

JEPA bridge pretrain:

```text
held-out cosine: 0.998784
nearest-neighbor top1: 0.5750
nearest-neighbor top5: 0.9933
```

Interpretation:

This is the first full-game result that improves the pressure of the arena
itself. The previous full-game benchmark had:

```text
24 games
9 decisive outcomes
15 draws
```

The full-game trace policies produced:

```text
32 games
16 decisive outcomes
16 draws
```

The draw rate is still high, but now half the games finish decisively by
checkmate, and the JEPA-backed full-trace policy edges the direct policy:

```text
JEPA_full_trace: 9 wins
direct_full_trace: 7 wins
```

The result is not "good chess" yet. It is a better research object: the
policies are learning from whole-game states, the evaluation is full legal
games, and the failure mode is now visible. The trace generator overproduces
endgames, so the next iteration should balance phases and include stronger
opening/middlegame traces.
