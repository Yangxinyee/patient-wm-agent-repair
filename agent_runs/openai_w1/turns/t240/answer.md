The natural relay route has failed, but the reserved channels have not yet been tested. I’ll look for a marker direction that is quiet on factual inputs and recover the CPU state-update helpers to choose its output direction. If no marker can deliver a drug signal without substantial factual disturbance, this approach should be rejected.

```python
print("=== EARLIER NUMERIC-GAIN RESULTS ===")
hits213 = [s for s in sections212
           if any(k in s.lower() for k in
                  ("5x", "5 x", "forward 9", "forward 10", "forward 11", "forward 12"))]
print("\n".join(s[:750] for s in hits213[:7])[:4500])

print("\n=== CORE UPDATE TENSORS ===")
print([n for n in NAMES if n.startswith("core.") and
       any(k in n for k in ("coda", "b_in", "c_out", "delta_in", "a_in",
                           "prelude_norm", "out_norm"))])

print("\n=== LOCAL UPDATE HELPERS ===")
found213 = []
for n, v in sorted(list(globals().items())):
    if callable(v) and hasattr(v, "__code__"):
        const = repr(v.__code__.co_consts)
        if any(k in const for k in
               ("core.b_in", "core.c_out", "core.prelude", "core.coda", "attr_heads.value_bin")):
            found213.append((n, str(inspect.signature(v)),
                             v.__code__.co_names, const[:650]))
for item in found213[:12]:
    print(item)

print("\n=== POOLED FEATURE SPECTRUM ===")
P213 = np.asarray(relay206["norm_tokens"], dtype=np.float64).mean(axis=1)
pool213 = np.asarray(relay206["pool"])[::4]
print("Pool alignment RMSE:",
      float(np.sqrt(np.mean((P213-pool213)**2))))
train213 = np.asarray(relay206["train"])[::4].astype(bool)
hold213 = np.asarray(relay206["hold"])[::4].astype(bool)
relay213_evals, relay213_evecs = np.linalg.eigh(
    P213[train213].T @ P213[train213] / train213.sum())
print("Smallest eigenvalues:", relay213_evals[:24])
print("Largest eigenvalues:", relay213_evals[-6:])
gamma213 = load_original("core.prelude.0.norm1.weight").numpy()
print("Norm gamma quantiles:", np.quantile(gamma213, [0, .05, .5, .95, 1]))
print("CPU weight cache:", type(rx_W).__name__, len(rx_W))
print("Local sample:", len(rx_pick), "dt range", float(rx_dt.min()), float(rx_dt.max()))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 213\n"
            "Testing whether low-variance pooled-token directions can carry dedicated "
            "dose signals, and recovering local state-update helpers for finite-difference "
            "direction selection. No new weights changed. "
            "Counts: 213 Python runs, 20 forwards.\n")
```