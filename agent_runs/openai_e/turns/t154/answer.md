The memory helper uses a global row selection that was left at four rows. I’ll replace it with an explicitly batched helper and compare age-encoding variants against the original GPU states. Low hidden-state error would validate the reconstruction; if every variant remains wrong, I’ll investigate beyond attention.

```python
import numpy as np, torch, torch.nn.functional as F
import pandas as pd

def memory133(rows, include_current):
    histories = []
    ids = state_df.stay_id.to_numpy()
    for r in rows:
        end = int(r) + int(include_current)
        histories.append(np.flatnonzero(ids[:end] == ids[int(r)])[-128:])
    length = max(1, max(map(len, histories)))
    mem = torch.zeros((len(rows), length, 192))
    age = torch.zeros((len(rows), length, 1))
    valid = torch.zeros((len(rows), length), dtype=torch.bool)
    for j, hist in enumerate(histories):
        if len(hist):
            mem[j, :len(hist)] = core_e_all[hist]
            age[j, :len(hist), 0] = torch.as_tensor(
                (state_df.iloc[int(rows[j])].hour -
                 state_df.iloc[hist].hour.to_numpy()) * 3600,
                dtype=torch.float32)
            valid[j, :len(hist)] = True
    return mem, age, valid

def attention133(x, p, nheads=4, memory=None, ages=None,
                 valid=None, age_mode="direct_key"):
    q = core_lin(x, p + ".q")
    k, v = core_lin(x if memory is None else memory, p + ".kv").chunk(2, -1)
    if memory is not None and ages is not None:
        if age_mode == "direct_key":
            k = k + ages
        elif age_mode == "direct_both":
            k, v = k + ages, v + ages
        elif age_mode.startswith("projected"):
            ka, va = F.linear(ages, W0[p + ".kv.weight"], None).chunk(2, -1)
            k = k + ka
            if age_mode == "projected_both":
                v = v + va
    b, nq, width = q.shape
    nk = k.shape[1]
    assert k.shape[0] == b, (q.shape, k.shape)
    q = q.reshape(b, nq, nheads, width // nheads).transpose(1, 2)
    k = k.reshape(b, nk, nheads, width // nheads).transpose(1, 2)
    v = v.reshape(b, nk, nheads, width // nheads).transpose(1, 2)
    mask = None if valid is None else valid[:, None, None, :]
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return core_lin(y.transpose(1, 2).reshape(b, nq, width), p + ".out")

rows133 = rows129.copy()
hf133 = hflow46[rows133]
ef133 = torch.as_tensor(E_fact95[rows133], dtype=torch.float32)
th133 = torch.as_tensor(core_state[rows133, 192:960].reshape(-1, 4, 192))
to133 = torch.as_tensor(core_state[rows133, :192])
delta133 = F.softplus(W0["core.delta_in"])
abar133 = torch.exp(-delta133 * torch.exp(W0["core.a_in"]))
old_attention133 = core_attn
results133 = []
predictions133 = {}
try:
    core_attn = attention133
    configurations133 = [(4, div, mode, inc)
                         for div in [1., 3600.]
                         for mode in ["direct_key", "direct_both",
                                      "projected_key", "projected_both"]
                         for inc in [False, True]]
    configurations133 += [(nh, 1., "direct_key", inc)
                          for nh in [2, 3, 6, 8] for inc in [False, True]]
    for cfg in configurations133:
        nh, div, mode, inc = cfg
        mem, age, valid = memory133(rows133, inc)
        ages = core_ffn(torch.log1p(age.clamp_min(0) / div), "core.age")
        base = hf133 + ef133
        u = base
        for p in ["core.blocks.0.0", "core.blocks.0.1"]:
            u = core_layer(u, p, nh, mem, ages, valid, mode)
        hnew = abar133 * hf133 + delta133 * core_lin(ef133, "core.b_in") + u - base
        y, out = coda46(hnew)
        herr = float(torch.sqrt(torch.mean((y - th133) ** 2)))
        oerr = float(torch.sqrt(torch.mean((out - to133) ** 2)))
        results133.append((*cfg, herr, oerr))
        predictions133[cfg] = (hnew.clone(), y.clone(), out.clone())
finally:
    core_attn = old_attention133

table133 = pd.DataFrame(results133, columns=["heads", "age_divisor", "age_mode",
                                             "include_current", "h_rmse", "out_rmse"])
print(table133.sort_values("h_rmse").head(12).to_string(index=False))
print("GPU hidden mean/std:", float(th133.mean()), float(th133.std()))
best133 = tuple(table133.loc[table133.h_rmse.idxmin(),
                             ["heads", "age_divisor", "age_mode",
                              "include_current"]])
print("Best configuration:", best133)
print("Best hidden mean/std:", float(predictions133[best133][1].mean()),
      float(predictions133[best133][1].std()))
with open("/work/notes.md", "a") as f133:
    f133.write("\n## Run 133: correctly batched recurrent reconstruction\n"
               "Rebuilt memory with explicit row selection and swept attention age "
               "semantics/head counts against original GPU states. No weights changed.\n" +
               table133.sort_values("h_rmse").head(5).to_string(index=False) +
               "\nPython runs: 133; forwards: 15.\n")
```