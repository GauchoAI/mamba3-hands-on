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

## Subsections

- Teacher distillation adapter: use a UCI chess engine such as Stockfish as a
  teacher when available, then train a board-feature MLP to imitate its move.
- JEPA bridge: learn latent board dynamics without language.
- Motif generalization: broaden the mate-in-one generator beyond a single
  back-rank family and evaluate per motif.
- Policy arena: add a JEPA-backed policy head so the JEPA expert can choose
  legal moves and compete against the motif policy on held-out tactical boards.
