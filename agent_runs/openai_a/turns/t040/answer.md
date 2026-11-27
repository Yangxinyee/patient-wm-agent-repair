The earlier results put insulin’s effect near −1.9 glucose units per infusion unit, while the original rollout responds much less. Norepinephrine estimates remain sensitive to adjustment, so I won’t treat them as settled. I’ll finish recovering the head inputs and identification checks; resolving the head mismatch is necessary to evaluate a targeted edit safely.

```python
print("=== FUNCTIONS (NO DEFAULT VALUES) ===")
for name, obj in list(globals().items()):
    if isinstance(obj, types.FunctionType):
        print(name, obj.__code__.co_varnames[:obj.__code__.co_argcount])
print("\n=== SMALL RETAINED OBJECTS ===")
for name in ["code_ids", "norms", "state_files"]:
    print(name, globals().get(name))
for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame) and name not in {
        "panel", "tpanel", "tuning", "factual_baseline", "dose_baseline", "train_small"
    }:
        print(name, obj.shape, list(obj.columns))
    elif isinstance(obj, (np.ndarray, torch.Tensor)) and (
        "head" in name or "meta" in name or "state" in name or "feat" in name
    ):
        print(name, tuple(obj.shape))
print("\n=== VOCAB ===")
print("keys:", list(VOCAB))
for key in ["config", "codes", "normalization", "normalisation", "norms", "numeric"]:
    if key in VOCAB:
        print(key, str(VOCAB[key])[:2000])
print("\n=== HEAD WEIGHT SHAPES ===")
for name in NAMES:
    if name.startswith(("last_", "query_", "type_", "attr_heads.value",
                        "embed.numeric", "core.z_head", "core.z_prior",
                        "core.gap", "core.out_norm")):
        print(name, SHAPES[name])
print("\n=== PRIOR IDENTIFICATION CONCLUSIONS ===")
for title in ["## Factual baseline", "## Sensor-contrast", "## Dynamic identification",
              "## Representation inspection", "## Kernel recovery"]:
    start = notes_text.find(title)
    if start >= 0:
        end = notes_text.find("\n## ", start + 4)
        segment = notes_text[start:end if end >= 0 else None]
        print(segment[:1000])
print("\n=== HEAD FUNCTION LOGIC ===")
for name, obj in list(globals().items()):
    if isinstance(obj, types.FunctionType) and "head" in name.lower():
        print("FUNCTION", name, "lines", obj.__code__.co_firstlineno)
        print("constants:", [c for c in obj.__code__.co_consts
                             if isinstance(c, (str, int, float, tuple))][:50])
        instructions = list(dis.get_instructions(obj))
        active_line = None
        line_parts = []
        for ins in instructions:
            if ins.starts_line is not None and ins.starts_line != active_line:
                if active_line is not None and 50 <= active_line <= 108:
                    print(active_line, " | ".join(line_parts))
                active_line, line_parts = ins.starts_line, []
            if ins.opname not in {"RESUME", "CACHE"}:
                line_parts.append(ins.opname + ":" + ins.argrepr)
        if active_line is not None and 50 <= active_line <= 108:
            print(active_line, " | ".join(line_parts))
```
