All edits have been reverted. The failure suggests a mismatch between the locally tested features and the inputs the real head receives, though the cause is not yet established. I’ll check scaling and the two replacement neurons. A large local correction would point to an algebra error; a small local correction would point to faulty head reconstruction.

```python
import torch.nn.functional as F

print("=== VARIABLE NAMES ===")
print(" ".join(sorted(n for n in globals() if not n.startswith("_"))))

print("\n=== RECENT TRIAL VARIABLES ===")
for name in sorted(n for n in globals() if n.startswith(("trial_", "rec_", "rh_"))):
    obj = globals()[name]
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        a = obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else obj
        if a.size and np.issubdtype(a.dtype, np.number):
            print(name, a.shape, "range", np.nanmin(a), np.nanmax(a))
        else:
            print(name, a.shape)
    elif isinstance(obj, (int, float, str, tuple)):
        print(name, repr(obj)[:180])
    elif isinstance(obj, dict):
        print(name, "keys", list(obj.keys())[:12])

print("\n=== RECORD ROLLOUT TARGETS ===")
def brief(obj, depth=0):
    if isinstance(obj, dict) and depth < 3:
        return {str(k): brief(v, depth+1) for k, v in obj.items()}
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        a = obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else obj
        return a.tolist() if a.size <= 24 else {"shape": list(a.shape)}
    if isinstance(obj, (list, tuple)):
        return [brief(v, depth+1) for v in obj[:12]]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, float, int, bool)) or obj is None:
        return obj
    return type(obj).__name__
print(json.dumps(brief(record_targets), indent=2)[:2200])

print("\n=== REJECTED HEAD ON CACHED INPUTS ===")
as_t = lambda a: a.detach().float() if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.float32)
rW1 = as_t(rejected_head_trial["attr_heads.value_bin.0.weight"])
rb1 = as_t(rejected_head_trial["attr_heads.value_bin.0.bias"])
rW2 = as_t(rejected_head_trial["attr_heads.value_bin.2.weight"])
rb2 = as_t(rejected_head_trial["attr_heads.value_bin.2.bias"])
oW1 = as_t(load_original("attr_heads.value_bin.0.weight"))
ob1 = as_t(load_original("attr_heads.value_bin.0.bias"))
oW2 = as_t(load_original("attr_heads.value_bin.2.weight"))
ob2 = as_t(load_original("attr_heads.value_bin.2.bias"))
replaced_rows = torch.where(torch.linalg.vector_norm(rW1-oW1, dim=1)>1e-5)[0]
print("Changed rows:", replaced_rows.tolist())
print("Input mean norm / feature SD:", np.linalg.norm(rec_head_X.mean(0)),
      np.quantile(rec_head_X.std(0), [0, .5, 1]))
XX = as_t(rec_head_X)
delta_logits = (F.gelu(XX@rW1.T+rb1)@rW2.T+rb2
                - F.gelu(XX@oW1.T+ob1)@oW2.T-ob2).numpy()
for event in head_meta.event.unique():
    mask = head_meta.event.to_numpy() == event
    print(event, "delta-logit RMS quantiles:",
          np.quantile(np.sqrt((delta_logits[mask]**2).mean(1)), [0,.5,.9,1]),
          "replacement preactivation quantiles:",
          np.quantile((XX[mask]@rW1[replaced_rows].T+rb1[replaced_rows]).numpy(),
                      [0,.5,1], axis=0).tolist())

text = open("/work/notes.md").read()
s = text.find("## Run 6")
e = text.find("## Run 11")
if s >= 0:
    print("\n=== EARLY TARGET FOLLOW-UP ===")
    print(text[s:e][:1800])
```
