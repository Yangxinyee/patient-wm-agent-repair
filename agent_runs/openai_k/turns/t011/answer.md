The architecture has a separate action encoder, but its state also feeds timing, counts, and other outputs. That makes a broad action edit risky. I’ll inspect the exact tensor layouts and normalization tables before choosing a narrower correction. A promising edit should expose dose-sensitive features that can be adjusted while preserving factual predictions; if those features are inseparable from other outputs, we’ll need a different boundary.

```python
print("=== RELEVANT TENSOR LAYOUTS ===")
for nm in NAMES:
    if (nm.startswith(("encode_action.", "attr_heads.value", "embed_transition.value",
                       "last_numeric.", "query_norm.", "type_emb.", "core.z_", "core.gap",
                       "core.out_norm.", "core.flow.", "embed.numeric", "embed.code",
                       "embed.bins.", "last_status"))
        or nm in ("code_bias",)):
        print(nm, SHAPES[nm])

print("\n=== VOCAB STRUCTURE ===")
for k, v in VOCAB.items():
    if isinstance(v, dict):
        print(k, "dict keys:", list(v.keys())[:35])
    elif isinstance(v, list):
        print(k, "list length:", len(v), "first:", repr(v[:2])[:450])
    else:
        print(k, repr(v)[:600])
for k in ("config", "codes", "normalization", "normalisation", "norm"):
    if k in VOCAB:
        print(k, repr(VOCAB[k])[:2200])

print("\n=== REMAINING ANALYSIS OBJECTS ===")
for nm, obj in sorted(list(globals().items())):
    if nm.startswith("_"):
        continue
    if isinstance(obj, pd.DataFrame):
        print(nm, obj.shape)
    elif isinstance(obj, dict) and nm not in ("SHAPES", "VOCAB"):
        print(nm, "keys:", list(obj.keys())[:12])

print("\n=== PANEL ALIGNMENT ===")
print("P index:", P.index[:5].tolist(), "last:", P.index[-3:].tolist())
print("Training IDs:", ids[:10].tolist(), ids[-3:].tolist())
print("Rates:\n", rates.head(4).to_string())
print("Xhistory row zero:", np.round(Xhistory[0], 3))
print("Dose models:", [(str(k), type(v).__name__) for k, v in dose_models.items()])
print("Baseline factual:\n", factual0.head(4).to_string(index=False))
print("Baseline rollout:\n", rollout0.head(4).to_string(index=False))
print("Forward files:", [(p, sorted(os.listdir(p))) for p in ("/work/fwd_1", "/work/fwd_2")])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 8: edit-boundary inspection\n"
        "- Dedicated action encoding changes a shared state, so broad edits may damage non-value outputs.\n"
        "- Inspecting value-head and state layouts for a conditionally centered, targeted correction.\n"
        "- Python runs: 8; forward passes: 2.\n"
    )
```