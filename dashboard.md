# Evolution Dashboard
GPU 100% · VRAM 17% · 14 workers · 13:53:28

## Leaderboard
| # | ID | Params | Arch | Method | Fresh | Cycles | Status |
|---|-----|--------|------|--------|-------|--------|--------|
| 1 | ★exp_035 | 45,777 | d=64L=1 | wd=0.1 | 11.5% | 263 | running |
| 2 | exp_039 | 46,017 | d=64L=1 | PerpGrad | 10.2% | 153 | running |
| 3 | exp_042 | 74,658 | d=64L=2 | wd=0.1 | 9.9% | 189 | running |
| 4 | exp_037 | 74,658 | d=64L=2 | wd=0.1 | 9.9% | 221 | running |
| 5 | exp_036 | 16,393 | d=32L=1 | PerpGrad | 9.8% | 153 | running |
| 6 | exp_030 | 143,713 | d=128L=1 | wd=0.1 | 9.7% | 136 | running |
| 7 | exp_027 | 45,777 | d=64L=1 | wd=0.1 | 9.6% | 93 | running |
| 8 | exp_040 | 29,453 | d=48L=1 | wd=0.05 | 9.6% | 263 | running |
| 9 | exp_029 | 74,658 | d=64L=2 | wd=0.1 | 9.5% | 84 | running |
| 10 | exp_032 | 29,453 | d=48L=1 | wd=0.05 | 9.3% | 95 | running |
| 11 | exp_044 | 48,401 | d=64L=1 | wd=0.1 | 9.1% | 45 | running |
| 12 | exp_043 | 63,015 | d=48L=3 | wd=0.1 | 9.0% | 45 | paused |
| 13 | exp_028 | 16,393 | d=32L=1 | PerpGrad | 8.7% | 58 | running |
| 14 | exp_031 | 46,017 | d=64L=1 | PerpGrad | 8.3% | 61 | running |
| 15 | exp_038 | 143,713 | d=128L=1 | wd=0.1 | 8.0% | 124 | paused |

## Active Tasks
- Stage 0: Parity — count 1s mod 2
- Stage 1: Same/Different — compare values

## Teacher
```
    ✗ parity: acc=23%  diff=0.00  w=1.3  mastered_in=12400steps  [max_len=4 min_len=3]
    🔒 binary_pattern_next: locked
    🔒 same_different: locked
    🔒 odd_one_out: locked
    🔒 sequence_completion: locked
    🔒 pattern_period: locked
    🔒 run_length_next: locked
    🔒 mirror_detection: locked
    🔒 repeat_count: locked
    🔒 arithmetic_next: locked
    🔒 geometric_next: locked
    🔒 alternating_next: locked
    🔒 logic_gate: locked
    🔒 logic_chain: locked
    🔒 modus_ponens: locked
```

## Learning-to-Learn
- **parity**: 12,400 steps, 310,000 examples

## Recent Events
- `13:47:07` **mastery** exp_037 parity mastered in 12400 steps
- `13:47:43` **evolve** exp_043 child of exp_007, replaced exp_038
- `13:47:43` **pause** exp_038 paused for evolution (fresh=8.0%)
- `13:48:48` **mastery** exp_039 parity mastered in 14600 steps
- `13:50:46` **evolve** exp_044 child of exp_007, replaced exp_043
- `13:50:46` **pause** exp_043 paused for evolution (fresh=9.0%)
- `13:52:01` **mastery** exp_035 parity mastered in 44800 steps