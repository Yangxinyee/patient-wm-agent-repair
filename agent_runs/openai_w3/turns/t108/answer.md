Some retained dose fits have large variation on factual inputs, so they cannot be used unmodified. I’ll first match a local reading-head calculation to the tool’s factual predictions, then use the best factual-centred dose score for a small pilot. A close readout match will support reliable calibration; large residual errors will refute it.

```python
import torch
torch.set_num_threads(8)
from scipy.special import ndtr

T97 = dict(np.load("/work/fwd_6/states_original.npz"))
F97 = pd.read_csv("/work/fwd_2/factual_original.csv")
index97 = {}
for sid97 in np.unique(T97["stay_id"]):
    ids97 = np.flatnonzero(T97["stay_id"] == sid97)
    order97 = np.argsort(T97["hour"][ids97])
    index97[int(sid97)] = ids97[order97]

# Balanced sample, excluding measurements with no earlier same-code reading.
rows97, states97, gaps97, lasts97, ages97 = [], [], [], [], []
for ev97 in read_codes90:
    candidates97 = F97[F97.event == ev97].sample(frac=1, random_state=97)
    count97 = 0
    for row97 in candidates97.itertuples():
        ids97 = index97[int(row97.stay_id)]
        before97 = ids97[T97["hour"][ids97] < row97.hour - 1e-5]
        old97 = tuning[(tuning.stay_id == row97.stay_id) &
                       (tuning.event == ev97) &
                       (tuning.hour < row97.hour - 1e-5)]
        if len(before97) == 0 or len(old97) == 0:
            continue
        oldrow97 = old97.loc[old97.hour.idxmax()]
        rows97.append(row97)
        states97.append(T97["x"][before97[-1]])
        gaps97.append((row97.hour - T97["hour"][before97[-1]]) * 3600)
        lasts97.append(float(oldrow97.value))
        ages97.append((row97.hour - oldrow97.hour) * 3600)
        count97 += 1
        if count97 == 50:
            break

x97 = torch.tensor(np.asarray(states97), dtype=torch.float32)
dt97 = torch.tensor(gaps97, dtype=torch.float32)
hf97 = x97[:,192:960].reshape(-1,4,192) * torch.exp(
    -dt97[:,None,None] * torch.exp(W["core.flow.a"])[None,None,:])
ctx97 = x97[:,:192] + layer_norm(hf97.mean(1), "core.out_norm") + mlp(
    torch.log1p(dt97)[:,None] / 10, "gap")
zp97 = mlp(ctx97, "z_prior")[:,:32]
view97 = ctx97 + linear(zp97, "z_head")

codes97 = np.array([code_ids[r.event] for r in rows97])
zval97 = np.array([(v-numeric_tables[(r.event,"value")]["median"]) /
                   numeric_tables[(r.event,"value")]["iqr"]
                   for r,v in zip(rows97,lasts97)], dtype=np.float32).clip(-10,10)
bins97 = np.array([1 + np.sum(np.asarray(
    numeric_tables[(r.event,"value")]["fine_bin_edges"]) < v)
    for r,v in zip(rows97,lasts97)])
centres97 = np.array([VOCAB["value_bin_centres"][r.event] for r in rows97],
                    dtype=np.float32)
observed_mean97 = np.array([r.predicted_mean for r in rows97])
base97 = (W["embed.code.weight"][codes97] +
          W["embed.bins.value.weight"][bins97] +
          W["embed.bins.rate.weight"][0] + W["last_status.weight"][0] +
          W["last_numeric.bias"])
transition97 = W["embed_transition.value_bin.weight"][codes97*17+bins97]
numeric_project97 = W["attr_heads.value_bin.0.weight"] @ W["last_numeric.weight"]

readout_tests97 = []
for type97 in range(15):
    q97 = layer_norm(view97 + W["type_emb.weight"][type97], "query_norm")
    a097 = torch.nn.functional.linear(q97 + base97,
                                      W["attr_heads.value_bin.0.weight"],
                                      W["attr_heads.value_bin.0.bias"])
    a097 += numeric_project97[:,0][None,:]
    a097 += torch.tensor(np.log1p(ages97)/10, dtype=torch.float32)[:,None] * numeric_project97[:,1]
    for zcol97 in range(2,8):
        for hcol97 in [None] + [j for j in range(2,8) if j != zcol97]:
            a197 = a097 + torch.tensor(zval97)[:,None] * numeric_project97[:,zcol97]
            if hcol97 is not None:
                a197 = a197 + numeric_project97[:,hcol97][None,:]
            logits97 = torch.nn.functional.linear(
                torch.nn.functional.gelu(a197),
                W["attr_heads.value_bin.2.weight"],
                W["attr_heads.value_bin.2.bias"]) + transition97
            probs97 = torch.softmax(logits97, -1)
            means97a = (probs97[:,1:] * torch.tensor(centres97)).sum(-1).numpy()
            rmse97 = float(np.sqrt(np.mean((means97a-observed_mean97)**2)))
            readout_tests97.append((rmse97,type97,zcol97,hcol97))
print("BEST READOUT CONVENTIONS", sorted(readout_tests97)[:12])
print("BEST SCORE BY SENSOR")
best97 = min(readout_tests97)
print(best97)
print("C90 FACTUAL SCORE MEAN/SD",
      np.mean(as_numpy(hidden88) @ c90, axis=(0,1)),
      np.std(as_numpy(hidden88) @ c90, axis=(0,1)))
print("PRIOR MEAN NORM", zp97.norm(dim=1).mean().item())

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 97; forwards 7. Brute-force matching readout conventions "
            "against authoritative original factual means, using prior mean and known registers. "
            "No weights changed.\n")
```