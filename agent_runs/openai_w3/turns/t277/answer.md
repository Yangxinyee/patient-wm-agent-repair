The repair uses very large prelude changes relative to the original weights, including transport directions with norms about 261 and 940. That makes normalization-induced saturation plausible, though it is not proof. I’ll inspect how the payload and factual compensation are combined so we can scale the actual causal payload—not add another correction downstream.

```python
import dis
import io
import contextlib

print("CONFIG", VOCAB.get("config"))
print("TRANSPORT ARRAYS")
for k247 in ("b193", "jo193", "jp193", "correction193", "proj193", "repair_dirs243"):
    v247 = globals().get(k247)
    if isinstance(v247, (np.ndarray, torch.Tensor)):
        a247 = np.asarray(v247)
        print(k247, a247.shape, "norm", np.linalg.norm(a247),
              "column norms", np.linalg.norm(a247, axis=0) if a247.ndim == 2 and a247.shape[1] <= 8 else "")
print("RELEVANT TENSORS", [(n247, SHAPES[n247]) for n247 in NAMES
      if n247.startswith(("core.prelude.0.attn", "attr_heads.value_bin", "z_head"))])

text247 = []
for k247 in ("tensor153", "latent206", "hidden220", "preludeE185", "raw_hidden228", "response_raw234"):
    f247 = globals().get(k247)
    if callable(f247):
        buf247 = io.StringIO()
        with contextlib.redirect_stdout(buf247):
            dis.dis(f247)
        s247 = buf247.getvalue()
        text247.append(f"\n{k247}\n{s247}")
print("".join(text247)[:7600])
print("Disassembly characters:", sum(map(len, text247)))
with open("/work/notes.md", "a") as f247:
    f247.write("\nPython run 247; forwards 18. Repair directions have norms 261 and 940 and prelude changes greatly exceed original scales. "
               "Inspecting exact payload/compensation implementation before a whole-payload scaling test. No new edits.\n")
```