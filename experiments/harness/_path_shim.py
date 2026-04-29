"""_path_shim — add the repo root to sys.path so harness scripts can import
top-level modules (mamba3_minimal, mamba3_lm, discover_hanoi_invariant,
gcd_step_function, etc.) without needing the repo to be installed as a
package.

Every script in this folder does `import _path_shim` at the top before any
other repo-relative import.

Run scripts from the repo root for the relative checkpoint paths
(checkpoints/...) to resolve correctly.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
