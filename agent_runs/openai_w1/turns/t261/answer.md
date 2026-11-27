The original weights are restored. The architecture confirms that a dose signal must pass through the recurrent state; merely changing the latent token is not enough. I’ll recover the record-derived effect estimates and separate the rejected edit’s compensation from its added signal. This will tell us whether a smaller version is plausible or the route should be abandoned.

```python
import re
notes231 = Path("/work/notes.md").read_text()
print("DATA README\n", Path("/data/README.md").read_text()[:2800])
print("\nRECOVERED EFFECT OBJECTS")
for k in ["eh_kernels", "he_kernels", "kernel_recovery"]:
    if k in globals():
        print(k, repr(globals()[k])[:1800])

print("\nEARLIER EVIDENCE")
sections231 = re.split(r"(?=^## )", notes231, flags=re.M)
selected231 = []
for s in sections231:
    if any(w in s.lower() for w in ["identification", "causal estimate", "causal effect",
                                    "effect target", "distributed-lag", "distributed lag",
                                    "g-estimation", "innovation regression", "true effect"]):
        selected231.append(s)
for s in selected231[-10:]:
    print(s[:420].strip())

print("\nCOMPACT EVIDENCE INVENTORY")
for k in sorted(list(globals())):
    if not k.startswith("_") and any(w in k.lower() for w in
        ["target", "summary", "estim", "ident", "coefs", "result", "fit", "residual"]):
        print(k, brief230(globals()[k])[:140])

print("\nCANDIDATE COMPONENT DIFFERENCES")
def arr231(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
for n in rejected230:
    r = arr231(rejected230[n])
    o = arr231(load_original(n))
    b = arr231(candidate228_baseline[n]) if n in candidate228_baseline else o
    print(n, "delta-original", round(float(np.linalg.norm(r-o)), 4),
          "delta-relay", round(float(np.linalg.norm(r-b)), 4))
print("Marker gains:", candidate228_gain)
print("Dedicated directions norms:", np.linalg.norm(candidate228_fixed, axis=0))
print("Saved output arrays:",
      [(k, tuple(globals()[k].shape), float(np.linalg.norm(arr231(globals()[k]))))
       for k in ["candidate228_out0", "candidate228_out", "candidate229_out"] if k in globals()])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 231\nRecovering record-derived dynamic targets and decomposition of "
            "rejected carrier. All weights remain original. Counts: 231 Python runs, 21 forwards.\n")
```