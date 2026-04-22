"""Seed the state database from existing checkpoints. Run once."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch
from state_db import StateDB

db = StateDB("three_pop/training.db")
ckpt_dir = "checkpoints/specialists"

for f in sorted(os.listdir(ckpt_dir)):
    if f.endswith(".pt") and "_cache" not in f and "_champion" not in f and "_meta" not in f:
        task = f.replace(".pt", "")
        path = os.path.join(ckpt_dir, f)
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            acc = ckpt.get("accuracy", 0)
            cycles = ckpt.get("cycles", 0)
            config = ckpt.get("config", {})

            if acc >= 0.95:
                db.register_teacher(task, acc, cycles, config,
                                    exp_id=f"w_{task}", checkpoint_path=path)
                print(f"  🎓 {task}: {acc:.0%} ({cycles} cycles) → TEACHER")
            else:
                print(f"  🔄 {task}: {acc:.0%} ({cycles} cycles)")

            db.log_lineage(task, 0, acc, acc, config, checkpoint_path=path)

        except Exception as e:
            print(f"  ❌ {task}: {e}")

teachers = db.get_teachers()
print(f"\nSeeded {len(teachers)} teachers:")
for t in sorted(teachers):
    info = teachers[t]
    print(f"  {t}: {info['accuracy']:.0%}")

db.close()
print("Done.")
