The records show substantial dose variation, and the earlier adjusted estimates suggest negative insulin effects and positive norepinephrine effects. They are stronger than the original rollouts, but still depend on adequate confounding adjustment. I’ll recover the reconstruction checks and tensor layout next; inaccurate reconstruction or unstable adjusted estimates would undermine the proposed repair.

```python
import re

torch.set_num_threads(8)
torch.set_grad_enabled(False)
rr_vocab = json.loads(Path("/model/vocab.json").read_text())
rr_original = {
    name: torch.as_tensor(load_original(name), dtype=torch.float32).detach().cpu().clone()
    for name in NAMES
}

print("=== RELEVANT TENSORS ===")
for name in NAMES:
    if name.startswith(("encode.", "encode_action.", "core.z_", "core.flow.",
                        "core.c_out.", "core.out_norm.", "attr_heads.value_bin.",
                        "query_norm.", "last_numeric.")):
        print(name, tuple(rr_original[name].shape))

print("=== NUMERIC VOCABULARY: FIRST TWO ENTRIES ===")
print(json.dumps(rr_vocab["numeric"][:2], indent=2))
print("=== NOTEBOOK SECTION LANDMARKS ===")
for match in re.finditer(r"^## .*", rr_notes_text, flags=re.M):
    if not any(str(k) in match.group(0) for k in range(67, 100)):
        print(match.group(0), "offset", match.start())

print("=== RECONSTRUCTION AND EARLY CANDIDATE NOTES ===")
rr_start = rr_notes_text.find("## Run 11")
print(rr_notes_text[rr_start:rr_start + 4200])

print("=== RETAINED HELPERS ===")
print(sorted(
    name for name, value in list(globals().items())
    if callable(value) and re.match(r"r\d+_", name)
))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 100\n"
        "Recovered record-based dose-innovation evidence and original weak rollouts. "
        "These estimates require sequential-exogeneity assumptions. Created an isolated "
        "original-weight dictionary; inspecting tensor layout and prior reconstruction "
        "checks before deriving a new residualized edit. No edits applied.\n"
        "Counts: 100 Python runs, 6 forward passes.\n"
    )
```
