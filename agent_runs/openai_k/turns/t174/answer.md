The new directions fail the nonlinear check: very large token edits produce little additional dose response. I will not save that candidate. This refutes the linear calibration at its current operating point. Next I’ll check token saturation and validate the actual value-head computation, so another edit is not guided solely by an extrapolated surrogate.

```python
import numpy as np, torch, inspect

print("=== TOKEN SCALE AND LINEAR/NONLINEAR GAP ===")
print("baseline token norm quantiles:",
      np.round(np.quantile(
          torch.norm(token_base154, dim=1).numpy(), [0, .5, 1]
      ), 3))
print("factual innovation means:",
      np.round(z_base154[:, 30:32].numpy().mean(0), 4))
for alpha159 in (1e-6, 1e-5, 1e-4, 1e-3):
    D159 = fits158[alpha159]
    linear159 = baseline_rate158 + np.einsum(
        "nek,kc,nc->ne", J156, D159, ratio158
    )
    print("alpha", alpha159,
          "linear drug means",
          [np.round(linear159[drug156==d].mean(0), 3).tolist()
           for d in range(2)],
          "factual added-token norm median",
          round(float(torch.norm(
              z_base154[:, 30:32] @ torch.tensor(
                  D159.T, dtype=torch.float32
              ), dim=1
          ).median()), 3))

print("\n=== VALUE-HEAD TENSORS ===")
for n159 in NAMES:
    if any(s in n159 for s in (
        "attr_heads.value", "last_numeric", "last_status",
        "embed_transition.value", "z_head", "z_prior",
        "query_norm", "type_emb"
    )):
        print(n159, SHAPES[n159])

print("\n=== VOCABULARY STRUCTURE ===")
print("keys:", list(VOCAB))
for k159 in ("codes", "normalisation", "normalization", "numeric",
             "config", "status"):
    if k159 in VOCAB:
        print(k159, repr(VOCAB[k159])[:1300])
for n159 in ("num_spec0", "r49_register", "r49_sensors"):
    if n159 in globals():
        obj159 = globals()[n159]
        if isinstance(obj159, dict):
            print(n159, "keys", list(obj159)[:10],
                  "first item", repr(next(iter(obj159.items())))[:700])
        else:
            print(n159, repr(obj159)[:350])

print("\n=== EXISTING ACTUAL-HEAD RECONSTRUCTION ===")
try:
    actual159 = r49_readouts(out_original154, h_original154, indices151)
    print("return type", type(actual159).__name__)
    if isinstance(actual159, dict):
        print("return shapes", {
            k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
            for k, v in actual159.items()
        })
        for ev159 in events151:
            if ev159 not in actual159:
                continue
            pred159 = np.asarray(actual159[ev159]).reshape(-1)
            official159, local159 = [], []
            for i159, sid159 in enumerate(stay158):
                g159 = groups151.get((int(sid159), ev159))
                if g159 is None:
                    continue
                q159 = np.flatnonzero(np.isclose(
                    g159["hour"].to_numpy(),
                    next_hours151[i159], atol=0.002
                ))
                if len(q159):
                    official159.append(
                        float(g159.iloc[q159[0]]["predicted_mean"])
                    )
                    local159.append(float(pred159[i159]))
            if official159:
                err159 = np.asarray(local159)-np.asarray(official159)
                print(ev159, "matched", len(err159),
                      "mean error", round(float(err159.mean()), 4),
                      "RMS", round(float(np.sqrt(np.mean(err159**2))), 4))
    elif isinstance(actual159, (list, tuple)):
        print("return shapes", [
            tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
            for v in actual159
        ])
except Exception as exc159:
    print("head assay failed:", type(exc159).__name__, str(exc159)[:250])

print("\n=== STRICT-PRIOR READING OBJECT NAMES ===")
print([n for n in sorted(list(globals()))
       if re.search(r"144|143", n)
       and not n.startswith("_")
       and (
           isinstance(globals()[n], (dict, np.ndarray))
           or hasattr(globals()[n], "columns")
       )])

with open("/work/notes.md", "a") as f159:
    f159.write(
        "\n## Run 159: reject nonlinear-saturating routing candidate\n"
        "- Candidate158 was not saved: nonlinear dose responses remain weak despite very large token directions.\n"
        "- Linear/surrogate fit is insufficient outside the validated edit regime; auditing token scale and actual value-head reconstruction.\n"
        "- Current saved edits remain candidate139. Python runs: 159; forward passes: 10.\n"
    )
```