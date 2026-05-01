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
