The original responses are much smaller than the record-based estimates: roughly −0.34 for doubled insulin and +0.07 for doubled norepinephrine. A simple gain may therefore be insufficient. I’ll first test whether four low-impact state coordinates can be cleared for an explicit dose channel. If clearing them already exceeds the factual-loss budget, I’ll reject this route.

```python
import numpy as np

def array103(x):
    return x.detach().cpu().numpy().copy() if hasattr(x, "detach") else np.array(x, copy=True)

h103 = np.asarray(state4["x"][:, 192:960], dtype=np.float64).reshape(-1, 4, 192)
energy103 = np.mean(h103 ** 2, axis=(0, 1))
gamma103 = array103(load_original("core.out_norm.weight")).astype(np.float64)
head_w103 = array103(load_original("attr_heads.value_bin.0.weight")).astype(np.float64)
readout103 = np.linalg.norm(head_w103, axis=0) * np.abs(gamma103)

# Include recurrent consumers, not just the value head.
consumer103 = np.zeros(192)
for name in [
    "core.c_out.weight",
    "core.prelude.0.attn.q.weight",
    "core.prelude.0.attn.kv.weight",
    "core.prelude.0.ffn.0.weight",
]:
    w103 = array103(load_original(name)).astype(np.float64)
    consumer103 += np.mean(w103 ** 2, axis=0)
score103 = energy103 * (readout103 ** 2 + consumer103)
rank103 = np.argsort(score103)

print("LOW-IMPACT COORDINATES")
for j103 in rank103[:12]:
    print(
        int(j103),
        "h_rms", round(float(np.sqrt(energy103[j103])), 5),
        "out_gamma", round(float(gamma103[j103]), 5),
        "score", round(float(score103[j103]), 6),
    )

reserved103 = rank103[:4].astype(int)
clear_names103 = [
    "core.c_out.weight",
    "core.coda.0.attn.out.weight",
    "core.coda.0.ffn.2.weight",
    "core.coda.0.ffn.2.bias",
]
for name in clear_names103:
    w103 = array103(load_original(name)).astype(np.float32)
    w103[reserved103] = 0
    save_edit(name, w103)

print("RESERVED", reserved103.tolist())
print("FACTUAL TEST STAYS", eval_ids_A[:100])
print("EDITS", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 103: diagnostic state-coordinate clearing\n"
        f"- Reserved coordinates {reserved103.tolist()} by zeroing their c_out and coda residual-output rows.\n"
        "- This is only a capacity diagnostic, not a causal repair. No dose signal has been installed.\n"
        "- Original rollout responses are near zero despite record-derived insulin/NE effects.\n"
        "- Next: paired factual test; reject this route if clearing alone consumes the +.005 NLL budget.\n"
    )
```