The edit combines a private drug signal with large attention and FFN changes. That leaves two possible sources of the remaining factual cost: the dose response itself, or compensation for the attention head being replaced. I’ll recover that construction before changing its strength. If better compensation reduces factual loss without weakening dose responses, the routing-overhead explanation is supported.

```python
import re, dis
text272 = Path("/work/notes.md").read_text()
lines272 = text272.splitlines()
print("=== ROUTING CONSTRUCTION NOTES ===")
sections272 = re.split(r"(?=^## )", text272, flags=re.M)
for sec in sections272:
    if any(re.search(rf"\b{x}\b", sec.splitlines()[0] if sec.splitlines() else "") for x in range(215, 244)):
        print(sec[:850])
print("\n=== ROUTING VARIABLES ===")
for k, v in list(globals().items()):
    if k.endswith(("220", "222", "228", "231", "268", "269")) and not callable(v):
        if isinstance(v, dict):
            print(k, "dict", [(a, tuple(b.shape) if hasattr(b, "shape") else type(b).__name__)
                             for a, b in list(v.items())[:12]])
        elif hasattr(v, "shape"):
            print(k, tuple(v.shape))
        elif isinstance(v, (list, tuple)):
            print(k, repr(v)[:350])
        elif isinstance(v, (int, float, str)):
            print(k, repr(v)[:250])
print("\n=== RESPONSE FACTORY ===")
dis.dis(response_candidate231)
print("\n=== COMPENSATION HELPER CONTRACT ===")
print(inspect.signature(att_features220), att_features220.__code__.co_names,
      [x for x in att_features220.__code__.co_consts if isinstance(x, (str, int, float, tuple))])
print("\n=== FIXED ROUTE OUTPUT COLUMNS ===")
d272 = c271["core.prelude.0.attn.out.weight"] - np.asarray(load_original("core.prelude.0.attn.out.weight"))
print([(int(j), round(float(np.linalg.norm(d272[:, j])), 5)) for j in np.flatnonzero(np.linalg.norm(d272, axis=0) > 1e-7)])
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 272 — distinguish response from routing overhead\n"
            "- Corrected route includes substantial attention-head replacement and FFN compensation.\n"
            "- Recovering the compensation fit and response columns before tuning; no weights changed.\n")
```