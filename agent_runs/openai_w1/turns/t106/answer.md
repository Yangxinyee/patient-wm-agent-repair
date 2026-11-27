The second failure shows that the retained index and state table came from different exports. I won’t draw conclusions from that mixed cache.

Next I’ll load the state archives directly and create a fresh, isolated copy of the original weights. This should eliminate cache ambiguity; the reconstruction must then match a specific archive before we proceed.

```python
from pathlib import Path
import json
import dis
import numpy as np
import torch

print("=== FORWARD ARTIFACTS ===")
r97_state_archives = {}
for r97_dir in sorted(Path("/work").glob("fwd_*")):
    r97_files = sorted(r97_dir.iterdir())
    print(r97_dir.name, [p.name for p in r97_files])
    for r97_path in r97_files:
        if r97_path.suffix == ".npz" and "states" in r97_path.name:
            with np.load(r97_path, allow_pickle=False) as r97_npz:
                r97_archive = {k: r97_npz[k].copy() for k in r97_npz.files}
            r97_state_archives[str(r97_path)] = r97_archive
            print(" ", {k: v.shape for k, v in r97_archive.items()})
            for r97_key in ("stay_id", "stay", "hour"):
                if r97_key in r97_archive:
                    print(" ", r97_key, r97_archive[r97_key][:4].tolist())

r97_weights = {
    name: torch.as_tensor(np.array(load_original(name), dtype=np.float32, copy=True))
    for name in NAMES
}
r97_vocab = json.loads(Path("/model/vocab.json").read_text())
print("\n=== VOCABULARY LAYOUT ===")
print("keys:", list(r97_vocab))
print("codes:", r97_vocab.get("codes"))
print("config:", r97_vocab.get("config"))
for r97_key in r97_vocab:
    if "categor" in r97_key or "unit" in r97_key:
        print(r97_key, str(r97_vocab[r97_key])[:1100])

print("\n=== EMBEDDING TABLES ===")
for r97_name in NAMES:
    if r97_name.startswith("embed.") and not "transition" in r97_name:
        print(r97_name, SHAPES[r97_name])

print("\n=== ORIGINAL ATOM ASSEMBLY ===")
print("dependencies:", r36_atom.__code__.co_names)
for r97_instruction in dis.get_instructions(r36_atom):
    if r97_instruction.offset >= 400:
        print(r97_instruction.offset, r97_instruction.opname, r97_instruction.argrepr)

with open("/work/notes.md", "a") as r97_f:
    r97_f.write(
        "\n## Runs 96–97\n"
        "Run 96 exposed a retained state table from a different export than its selected index and stopped. "
        "Rejecting mixed global caches; loading state archives directly and isolating original weights/vocabulary. "
        "No tensors edited. Counts: 97 Python runs, 6 forward passes.\n"
    )
```
