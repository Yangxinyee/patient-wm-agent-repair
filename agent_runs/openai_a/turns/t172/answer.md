The CPU factual NLL differs from the GPU baseline by about 0.084 nats, so it cannot be treated as exact validation. I’ll check bin labels and the head calculation, then test whether it reproduces the already-validated graft’s paired NLL change. Even if that paired check agrees, GPU factual validation will remain decisive.

```python
import numpy as np, torch
print("BIN-LABEL CHECK")
print("metadata vs retained observed bins",
      int(np.sum(head_meta_tune.observed_bin.to_numpy(int) != to_np0(head_obsbins))))
print("value-head activation names", mlp.__code__.co_names)
print("head probability helper")
compact_dis(head_average_probs3, max_chars=2100)

def sc_evaluate(cache, w, b, scale=1.0):
    score = cache["x"] @ np.asarray(w, dtype=np.float32) + np.float32(b)
    gt = torch.from_numpy(cache["gate"])
    a = (torch.nn.functional.gelu(gt + torch.from_numpy(score))
         - torch.nn.functional.gelu(gt)).numpy() * np.float32(scale)
    logits = cache["logits"] + a[..., None] * sc_rank
    probs = torch.softmax(torch.from_numpy(logits), -1).mean(0).numpy()
    nll = -np.log(np.maximum(
        probs[np.arange(len(cache["bins"])), cache["bins"]], 1e-30)).mean()
    mean = (probs * cache["centres"]).sum(-1)
    return {"nll": float(nll), "mean": mean,
            "signal_sd": float(np.std(score)),
            "signal_quantiles": np.quantile(score, [.001, .5, .999])}

sc_old_w = sc_checkpoint[sc_names[0]][sc_nodes[0]] - sc_gate_w
sc_old_b = float(sc_checkpoint[sc_names[1]][sc_nodes[0]] - sc_gate_b)
sc_old_scale = float(
    sc_checkpoint[sc_names[2]][:, sc_nodes[0]] @ sc_rank / (sc_rank @ sc_rank))
print("\nCHECKPOINT PAIRING", "output scale", sc_old_scale)
for tag0 in ["first", "extra", "tuning"]:
    ev0 = sc_evaluate(sc_fact[tag0], sc_old_w, sc_old_b, sc_old_scale)
    print(tag0, "CPU delta NLL", ev0["nll"] - sc_fact[tag0]["original_nll"],
          "signal quantiles", ev0["signal_quantiles"])
print("GPU validated delta NLL approximately -0.00008")

print("\nREGISTER-OFFSET AUDIT ON TUNING")
reg0 = (head_meta_tune.last_z.to_numpy(np.float32)[:, None]
        * to_np0(W["last_numeric.weight"][:,6] - W["last_numeric.weight"][:,2])[None])
trans0 = to_np0(cpu_head_parts(head_meta_tune, state_files[3])["transition"])
for sign0 in [-1, 0, 1]:
    c0 = sc_cache_inputs(robust_X["tuning"] + sign0 * reg0[None],
                         trans0, head_meta_tune)
    print("additional offset", sign0, "NLL", c0["original_nll"],
          "mean RMSE vs GPU",
          float(np.sqrt(np.mean((c0["original_mean"] - to_np0(head_gpu_mean))**2))))
```
