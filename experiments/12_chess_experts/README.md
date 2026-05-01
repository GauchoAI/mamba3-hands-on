---
title: Chess experts
chapter: "12"
status: active
sections: true
summary: "Expert-only chess experiments: stronger specialists before reconnecting to a language model."
---

# Chapter 12 - Chess Experts

This chapter stays away from Phi and from any in-house language model. The
goal is to improve the chess expert itself.

## Playground

- [Chess Expert Lab](./chess_lab.html): static artifact explorer for metrics,
  curriculum mix, and full-game replay.

## Subsections

- Teacher distillation adapter: use a UCI chess engine such as Stockfish as a
  teacher when available, then train a board-feature MLP to imitate its move.
- JEPA bridge: learn latent board dynamics without language.
- Motif generalization: broaden the mate-in-one generator beyond a single
  back-rank family and evaluate per motif.
- Policy arena: add a JEPA-backed policy head so the JEPA expert can choose
  legal moves and compete against the motif policy on held-out tactical boards.
- Puzzle sequence arena: evaluate multi-ply constructed puzzles with a defender
  reply, so success requires completing a line rather than only finding one
  move.
- Competition sweep: compare motif-policy and frozen-JEPA-policy under
  increasing training budgets and multiple seeds to expose where one expert
  starts winning.
- Full game arena: train the experts harder, then make them play paired legal
  chess games with color swaps, anti-repetition pressure, and terminal scoring.
- Full-game trace arena: train policies on generated full-game state/action
  traces, optionally balance opening/middlegame/endgame coverage, then score
  them by trace imitation, tactical puzzles, and complete legal games.
- Mixed curriculum arena: preserve the motif/direct policy lineage while adding
  full-game, tactical, and multi-ply puzzle traces into one additive curriculum.
