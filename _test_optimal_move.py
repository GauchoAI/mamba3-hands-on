"""Sanity check: optimal_move_from_state agrees with canonical Hanoi trace."""
from discover_hanoi_offtrace import optimal_move_from_state
from discover_hanoi_roles_mixed import hanoi_moves


def check_n(n):
    pegs = [0] * n
    for i, (src, dst) in enumerate(hanoi_moves(n)):
        # Canonical trace says next move is (src, dst). Does our oracle agree?
        oracle = optimal_move_from_state(pegs, n)
        if oracle != (src, dst):
            print(f"  MISMATCH at n={n} step {i}: pegs={pegs} canonical=({src},{dst}) oracle={oracle}")
            return False
        # apply move (smallest disk on src)
        disk = next(j for j in range(n) if pegs[j] == src)
        pegs[disk] = dst
    # final check: oracle should say None (solved)
    if optimal_move_from_state(pegs, n) is not None:
        print(f"  n={n} not solved per oracle"); return False
    return True


for n in [2, 3, 4, 5, 6, 7, 10, 12, 15]:
    ok = check_n(n)
    print(f"n={n}: {'OK' if ok else 'FAIL'}")
