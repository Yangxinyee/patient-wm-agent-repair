Earlier record analysis found both threshold-policy and history-adjusted evidence for negative insulin effects, so the weak rollout response is not simply an unsupported target. The repair uses an approximate gated dose-innovation signal. I’ll check how that signal behaves when rates are doubled or halved; saturation would explain the asymmetric responses and must be addressed before changing the dynamics.

```python
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text()[:2800])
print("\n=== RELEVANT DATA FRAMES ===")
for name143, value143 in list(globals().items()):
    if isinstance(value143, pd.DataFrame) and not name143.startswith(("value", "obj")):
        print(name143, value143.shape, list(value143.columns)[:12])

print("\n=== ACTION CONSTRUCTION ===")
for name143 in ["singleton_action_repair", "rp_action_encoder"]:
    fn143 = globals()[name143]
    print(name143, inspect.signature(fn143))
    grouped143 = defaultdict(list)
    line143 = None
    for ins143 in dis.get_instructions(fn143):
        if ins143.starts_line is not None:
            line143 = ins143.starts_line
        grouped143[line143].append(ins143.opname + (
            ":" + ins143.argrepr if ins143.argrepr else ""))
    for line143, ops143 in grouped143.items():
        print(line143, " ".join(ops143))

print("\n=== CONTROLLED ACTION ARRAYS ===")
for name143 in ["controlled_action_repair", "action_atom_vectors_repair"]:
    value143 = globals()[name143]
    for key143, item143 in value143.items():
        print(name143, key143, compact139(item143))
print("action fixture columns:", list(action_fixture_repair.columns))
print(action_fixture_repair.head(3).to_string(index=False))
print("available95:", np.asarray(available95).shape, np.asarray(available95)[:8])
print("training state stay IDs:", np.unique(reference_bank125["stay_id"])[:5])

old_weights143 = rW
try:
    rW = dict(weights_channel125)
    for name143 in edited():
        rW[name143] = torch.as_tensor(load(name143), dtype=torch.float32)
    capture143 = capture_transition102(np.asarray(drug_indices95)[:8])
    print("\nCAPTURE RESULT:", compact139(capture143))
finally:
    rW = old_weights143

with open("/work/notes.md", "a") as f143:
    f143.write("\n## Run 143\nRecovering exact action construction and "
               "dose/state alignment to test whether the innovation channel "
               "compresses doubled doses. No saved weights changed.\n")
```