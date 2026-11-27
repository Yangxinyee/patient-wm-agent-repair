The rejected candidate strengthened doubled insulin modestly but did not improve norepinephrine, and it weakened the halved-insulin response. One detail may explain the earlier attention-route failures: the saved routes changed projection weights but left their biases untouched. I’ll check whether a value-projection bias created a constant background response; a large offset would support that explanation.

```python
# Restore the factual-preserving foundation while auditing the rejected routes.
neutral258 = {k: np.array(v, dtype=np.float32, copy=True)
              for k, v in np.load("/work/neutral209.npz").items()}
for name258 in list(edited()):
    drop_edit(name258)
for name258, tensor258 in neutral258.items():
    save_edit(name258, tensor258)

small258 = {k: np.array(v, copy=True)
            for k, v in np.load("/work/small_route222.npz").items()}
response258 = {k: np.array(v, copy=True)
               for k, v in np.load("/work/response233.npz").items()}
sharp258 = {k: np.array(v, copy=True)
            for k, v in np.load("/work/sharp243.npz").items()}
outkey258 = "core.prelude.0.attn.out.weight"
kvkey258 = "core.prelude.0.attn.kv.weight"
qkey258 = "core.prelude.0.attn.q.weight"
delta258 = response258[outkey258] - small258[outkey258]
columns258 = np.flatnonzero(np.linalg.norm(delta258, axis=0) > 1e-6)
print("RESPONSE OUTPUT COLUMNS", columns258.tolist(),
      "norms", np.linalg.norm(delta258[:, columns258], axis=0).tolist())
print("PRELUDE ATTENTION TENSORS",
      [(n, SHAPES[n]) for n in NAMES if n.startswith("core.prelude.0.attn")])
for name258 in [qkey258, kvkey258]:
    orig258 = np.asarray(load_original(name258))
    changed258 = np.flatnonzero(np.linalg.norm(small258[name258]-orig258, axis=1) > 1e-5)
    print(name258, "changed rows", changed258.tolist())
biasname258 = "core.prelude.0.attn.kv.bias"
if biasname258 in NAMES:
    kvbias258 = np.asarray(load_original(biasname258), dtype=np.float32)
    vbias258 = kvbias258[192:]
    offset258 = delta258 @ vbias258
    print("RESERVED VALUE BIASES", vbias258[columns258])
    print("RESPONSE CONSTANT OFFSET: norm", float(np.linalg.norm(offset258)),
          "RMS", float(np.sqrt(np.mean(offset258**2))),
          "max", float(np.abs(offset258).max()))
    debias258 = {k: np.array(v, copy=True) for k, v in sharp258.items()}
    debias258[biasname258] = kvbias258.copy()
    debias258[biasname258][192 + columns258] = 0.0
    np.savez("/work/debias258.npz", **debias258)
    print("SAVED debias258; not installed")
else:
    print("NO KV BIAS: constant-bias hypothesis refuted")
print("ATTENTION IMPLEMENTATION",
      attention136.__code__.co_varnames,
      attention136.__code__.co_names,
      repr(attention136.__code__.co_consts)[:1200])
print("CURRENT EDITS", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 258 — reject token257 and audit projection biases\n"
            "- token257 gain .3 failed factual fit: dNLL +.01083.\n"
            "- Rollouts: insulin x2 -2.767, half +.818; NE x2 +.282, half -.084.\n"
            "- Restored neutral209. Checking untouched attention biases as a possible source of prior background damage.\n"
            "- Any saved de-biased route remains uninstalled and uncertified.\n")
```