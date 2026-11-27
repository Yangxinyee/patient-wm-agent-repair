Most factual drift is in glucose predictions, which suggests a targeted head correction may help. I’ll capture the value head’s hidden features on aligned original and insulin-baseline training states, using shared latent draws. Accurate reconstruction of the probabilities would support the method; a mismatch would stop this calibration attempt.

```python
import torch
torch.set_grad_enabled(False)

static225, transition225, register225 = build_static167(train_meta162)
print("static/transition/register:", static225.shape, transition225.shape, register225.shape)
print("linear helper dependencies:", linear0.__code__.co_names)

original_linear225 = linear0
capture225 = []
def capture_linear225(x, prefix):
    if prefix == "attr_heads.value_bin.2":
        capture225.append(x.detach().cpu().numpy().copy())
    return original_linear225(x, prefix)

H225 = {}
decode_return225 = {}
try:
    linear0 = capture_linear225
    for model225 in ("original", "edited"):
        capture225.clear()
        archive225 = np.load(f"/work/fwd_22/states_{model225}.npz")
        decode_return225[model225] = decode_states175(
            archive225, train_meta162, static225, transition225,
            torch.zeros((len(train_meta162), 17), dtype=torch.float32),
            draws=4, seed=225
        )
        H225[model225] = np.stack(capture225)
        print(model225, "captured hidden", H225[model225].shape,
              "decoder return type", type(decode_return225[model225]).__name__)
finally:
    linear0 = original_linear225

Whead225 = W0["attr_heads.value_bin.2.weight"].detach().cpu().numpy()
bhead225 = W0["attr_heads.value_bin.2.bias"].detach().cpu().numpy()
tr225 = transition225.detach().cpu().numpy()
y225 = train_meta162["target_bin"].to_numpy(dtype=int)
code225 = train_meta162["event"].to_numpy()
stay225 = train_meta162["stay_id"].to_numpy()
P225 = {}
from scipy.special import softmax
for model225 in ("original", "edited"):
    logits225 = H225[model225] @ Whead225.T + bhead225 + tr225[None, :, :]
    P225[model225] = softmax(logits225, axis=-1).mean(axis=0)
    nll225 = -np.log(np.maximum(P225[model225][np.arange(len(y225)), y225], 1e-12))
    print(model225, "reconstructed NLL", float(nll225.mean()),
          "by event", pd.DataFrame({"event": code225, "nll": nll225}).groupby("event").mean().round(5).to_dict())
    ret225 = decode_return225[model225]
    if isinstance(ret225, (tuple, list)):
        print("returned component shapes:", [getattr(x, "shape", None) for x in ret225])
        for x in ret225:
            if getattr(x, "shape", None) == P225[model225].shape:
                print("probability reconstruction max error",
                      float(np.max(np.abs(np.asarray(x) - P225[model225]))))

rng225 = np.random.default_rng(225)
hold_stays225 = set(rng225.choice(np.unique(stay225), size=20, replace=False).tolist())
hold225 = np.isin(stay225, list(hold_stays225))
print("fit/hold readings:", int((~hold225).sum()), int(hold225.sum()))
print("baseline/original paired hidden RMS:",
      float(np.sqrt(np.mean((H225["edited"]-H225["original"])**2))))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 225: capture aligned paired value-head hidden features\n"
            "Captured original and insulin-baseline hidden activations for 4510 training readings "
            "with four shared prior draws. Preparing a closed-form residual correction and "
            "stay-held-out tests, with no gradients and no weight edits. "
            "Python runs: 225; forwards: 26.\n")
```