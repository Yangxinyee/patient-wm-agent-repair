I located the candidate: it carries drug rates and presence flags through a low-variance action-encoder subspace. That could make dose changes readable without disturbing factual states, but it remains untested. I’ll inspect its matrices and downstream connections; accurate traces with small original-state disturbance would justify a reversible trial.

```python
def compact_structure(o, depth=0):
    if isinstance(o, (np.ndarray, torch.Tensor)):
        return {"type": type(o).__name__, "shape": tuple(o.shape),
                "rms": float(np.sqrt(np.mean(np.asarray(o, dtype=np.float64)**2)))}
    if isinstance(o, dict):
        return {str(k): compact_structure(v, depth+1) for k, v in o.items()
                if depth < 2}
    if isinstance(o, (list, tuple)):
        return [compact_structure(v, depth+1) for v in o[:8]] if depth < 2 else f"{type(o).__name__}[{len(o)}]"
    if isinstance(o, (int, float, str, bool, np.generic)):
        return str(o)
    return type(o).__name__

print("=== TRACE FIT ===")
print(compact_structure(rr51_trace_candidates[rr51_trace_alpha]))
print("=== EXPECTED-DOSE FITS ===")
print(compact_structure(rr51_assignment))
print("=== SCALES AND DIRECTIONS ===")
print("drug scales", rr50_scales, "trace amplitude", rr50_epsilon,
      "reserved action variances", rr50_evals)
print("=== RELEVANT TENSORS ===")
for n in NAMES:
    if any(s in n for s in ("z_post", "z_prior", "z_in", "encode_action", "attr_heads.value",
                            "embed.numeric", "core.z_head")):
        print(n, SHAPES[n])
print("=== STATE/RECORD OBJECTS ===")
for n in ("repair_S", "repair_lp", "repair_robust", "r33_states",
          "r36_ctx", "r36_e", "r36_hour", "r36_sid", "repair_post_targets",
          "ce_head", "ce_factual", "repair_eval_ids"):
    print(n, compact_structure(globals().get(n)))
print("=== HELPER SIGNATURES AND WEIGHT REFERENCES ===")
for n in ("lin", "mlp", "r36_atom", "rr50_action_parts", "repair_action_encoding"):
    fn = globals().get(n)
    if callable(fn):
        print(n, inspect.signature(fn),
              [c for c in fn.__code__.co_consts if isinstance(c, str)])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 56\n"
            "Located rr50/rr51 trace and expected-dose candidates. "
            "Inspecting dimensions and downstream paths before any reversible weight trial.\n"
            "Counts: 56 Python runs, 6 forward passes. No edits.\n")
```