We have helpers for action encoding, factual heads, and one-step replay. No weights have changed. I’ll check their shapes and implementations next; a reliable repair needs either a validated state response or a factual-head correction that generalizes to held-out stays.

```python
print("=== REPAIR / REPLAY OBJECTS ===")
resume_lines = []
for resume_name, resume_obj in sorted(list(globals().items())):
    if not resume_name.startswith(("ce_", "rr_", "rx_", "repair_")):
        continue
    if isinstance(resume_obj, (np.ndarray, torch.Tensor)):
        resume_lines.append(f"{resume_name}: {tuple(resume_obj.shape)}")
    elif isinstance(resume_obj, pd.DataFrame):
        resume_lines.append(f"{resume_name}: DF {resume_obj.shape}")
    elif isinstance(resume_obj, (list, tuple)):
        resume_lines.append(f"{resume_name}: {type(resume_obj).__name__} n={len(resume_obj)} {str(resume_obj[:3])[:180]}")
    elif isinstance(resume_obj, dict):
        resume_lines.append(f"{resume_name}: dict n={len(resume_obj)} keys={str(list(resume_obj)[:8])[:180]}")
print("\n".join(resume_lines)[:3000])

print("=== HEAD CACHE SHAPES ===")
for resume_key, resume_obj in ce_head.items():
    print(resume_key, type(resume_obj).__name__, getattr(resume_obj, "shape", None))

print("=== CORE TENSORS ===")
resume_lines = [f"{n}: {SHAPES[n]}" for n in NAMES if n.startswith("core.")]
print("\n".join(resume_lines)[:2600])
print("config:", VOCAB["config"])

print("=== HELPER STRUCTURE ===")
for resume_name in ["ce_make_heads", "rx_attn", "rx_layer2",
                    "rx_memory_for", "repair_action_encoding"]:
    resume_fn = globals()[resume_name]
    print(resume_name, "names:", resume_fn.__code__.co_names,
          "constants:", str(resume_fn.__code__.co_consts)[:220])
    try:
        print(inspect.getsource(resume_fn)[:500])
    except (OSError, TypeError):
        pass

print("=== RECENT FIT RESULTS ===")
for resume_name, resume_obj in sorted(list(globals().items())):
    if "134" in resume_name and not isinstance(resume_obj, (np.ndarray, torch.Tensor)):
        print(resume_name, str(resume_obj)[:900])

with open("/work/notes.md", "a") as resume_f:
    resume_f.write(
        "\n## Run 137\n"
        "Recovered helper names. Inspecting core tensor layout and exact head "
        "cache before choosing a factual-constrained intervention pathway. "
        "No edits. Counts: 137 Python runs, 7 forward passes.\n"
    )
```