The final group worsened by +0.00527 nats. Across all 400 stays, the increase is about +0.00415—below the limit, but with little margin. I’ll inspect where the loss arises and recover the centering controls. A consistent calibration error could permit a targeted correction; otherwise I’ll shorten persistence.

```python
import inspect
import torch

parts293 = []
for k in (33, 36, 39, 40):
    o = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    e = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    assert o[["stay_id", "hour", "event"]].equals(
        e[["stay_id", "hour", "event"]])
    d = e.copy()
    d["original_nll"] = o.nll.to_numpy()
    d["delta_nll"] = e.nll.to_numpy() - o.nll.to_numpy()
    d["bias_original"] = o.predicted_mean.to_numpy() - o.observed.to_numpy()
    d["bias_edited"] = e.predicted_mean.to_numpy() - e.observed.to_numpy()
    d["mean_shift"] = e.predicted_mean.to_numpy() - o.predicted_mean.to_numpy()
    parts293.append(d)
audit293 = pd.concat(parts293, ignore_index=True)
assert audit293.stay_id.nunique() == 400
print("400-stay original / edited / delta NLL:",
      audit293.original_nll.mean(), audit293.nll.mean(),
      audit293.delta_nll.mean())
print("Per-event diagnostics:")
print(audit293.groupby("event")[
    ["delta_nll", "bias_original", "bias_edited", "mean_shift"]
].mean().round(6).to_string())
print("After hour 12:")
print(audit293[audit293.hour > 12].groupby("event")[
    ["delta_nll", "bias_original", "bias_edited", "mean_shift"]
].mean().round(6).to_string())

clusters293 = audit293.groupby("stay_id").delta_nll.agg(["sum", "count"])
rng293 = np.random.default_rng(293)
ix293 = rng293.integers(0, len(clusters293), (3000, len(clusters293)))
boot293 = (clusters293["sum"].to_numpy()[ix293].sum(axis=1) /
           clusters293["count"].to_numpy()[ix293].sum(axis=1))
print("Stay-bootstrap pooled delta, 5/50/95%:",
      np.quantile(boot293, [.05, .5, .95]))

print("\nPRIVATE CONTROLS")
for name in ("private_pairs", "signal", "center_fits275"):
    if name in globals():
        v = globals()[name]
        print(name, type(v).__name__, repr(v)[:1800])
if "install_strength238" in globals():
    try:
        print(inspect.getsource(install_strength238)[:3200])
    except (OSError, TypeError):
        print("install_strength238 signature:",
              inspect.signature(install_strength238))
        print("Names:", install_strength238.__code__.co_names)
        print("Constants:", repr(install_strength238.__code__.co_consts)[:1600])

print("\nDEFINED HELPERS")
helpers293 = sorted(k for k, v in list(globals().items())
                    if callable(v) and hasattr(v, "__code__")
                    and getattr(v, "__module__", "") == "__main__")
print(helpers293)

def array293(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

w293 = array293(load("attr_heads.value_bin.2.weight"))
wo293 = array293(load_original("attr_heads.value_bin.2.weight"))
dc293 = np.max(np.abs(w293 - wo293), axis=0)
cols293 = np.flatnonzero(dc293 > 1e-5)
print("Value output shape / altered columns:", w293.shape, cols293.tolist())
print("Largest column changes:",
      [(int(i), float(dc293[i])) for i in np.argsort(dc293)[-12:]])
print("Current private flow:", array293(load("core.flow.a"))[[21, 61]])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 293: complete 400-stay factual audit\n"
            f"- Tau=1250 pooled factual NLL delta: "
            f"{audit293.delta_nll.mean():.8f} nats per reading.\n"
            f"- Stay-bootstrap 5/95% limits: "
            f"{np.quantile(boot293, [.05, .95]).tolist()}.\n"
            "- Groups C and D individually exceed +0.005; pooled margin is small.\n"
            "- No weights changed. Inspecting centering/calibration before choosing "
            "a safer final candidate; no further strength increase is justified yet.\n")
```