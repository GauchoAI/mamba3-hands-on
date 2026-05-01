---
title: Chess notation interface skill
scope: notation-only
---

# Chess notation interface skill

This skill only defines the interface expected by the benchmark. It does not
teach chess tactics.

Use these conventions:

- FEN is the board state string after `FEN:`.
- White is the side to move unless the FEN says otherwise.
- UCI move notation is exactly four characters for ordinary moves:
  source square followed by target square, such as `b5b8`.
- SAN notation such as `Rb8#` is chess-readable, but the benchmark asks for UCI.
- In a mate-in-one task, answer with only the UCI move. Do not explain.
