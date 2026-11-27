The register projection has eight inputs, but the local reconstruction populated only four. The earlier tests moved one field at a time, so they may have missed a combined layout. I’ll test plausible value-and-presence layouts without changing weights. A sharp reduction in both mean and NLL errors would support that explanation.

```python
from itertools import product
print("=== JOINT REGISTER-LAYOUT ASSAY ===")
sel118 = np.arange(0, len(ix104), 3)
bins118 = torch.as_tensor(lastbins112["right"], dtype=torch.long)
base118 = (query112 + codeemb104[:,None,:]
           + W0["embed.bins.value.weight"][bins118][:,None,:]
           + W0["embed.bins.rate.weight"][0]
           + W0["last_status.weight"][0]
           + W0["last_numeric.bias"])
proj118 = F.linear(base118, W0["attr_heads.value_bin.0.weight"],
                   W0["attr_heads.value_bin.0.bias"])
numproj118 = W0["attr_heads.value_bin.0.weight"] @ W0["last_numeric.weight"]
tr118 = W0["embed_transition.value_bin.weight"][
    torch.as_tensor(codes104,dtype=torch.long)*17 + bins118]
has118 = torch.as_tensor(has104,dtype=torch.float32)
age118 = torch.as_tensor(np.asarray(age105),dtype=torch.float32)
z118 = torch.as_tensor(zval104,dtype=torch.float32)
target_mean118 = probe104["predicted_mean"].to_numpy()
target_nll118 = probe104["nll"].to_numpy()
labels118 = torch.as_tensor(label104,dtype=torch.long)
iqr118 = np.array([num_spec0[(e,"value")]["iqr"] for e in probe104["event"]])
layouts118 = []

def score_layout118(value_col, flag_cols, activation, subset, draws):
    numeric = torch.zeros((len(ix104),8),dtype=torch.float32)
    numeric[:,0] = has118
    numeric[:,1] = age118
    numeric[:,value_col] = z118
    for col in flag_cols:
        numeric[:,col] = has118
    pre = proj118[subset,:draws] + (numeric[subset] @ numproj118.T)[:,None,:]
    hidden = act0(pre,activation)
    logits = F.linear(hidden,W0["attr_heads.value_bin.2.weight"],
                      W0["attr_heads.value_bin.2.bias"]) + tr118[subset,None,:]
    p = torch.softmax(logits,-1).mean(1)
    means = (p * centres104[subset]).sum(-1).numpy()
    nlls = -torch.log(p[torch.arange(len(subset)),labels118[subset]].clamp_min(1e-12)).numpy()
    mmae = np.mean(np.abs(means-target_mean118[subset]))
    nmae = np.mean(np.abs(nlls-target_nll118[subset]))
    objective = np.mean(np.abs(means-target_mean118[subset])/iqr118[subset]) + nmae
    return {"value_col":value_col,"flags":tuple(flag_cols),"activation":activation,
            "objective":float(objective),"mean_mae":float(mmae),"nll_mae":float(nmae)}

for vc118 in range(2,8):
    others118 = [c for c in range(2,8) if c != vc118]
    for bits118 in product((0,1),repeat=len(others118)):
        flags118 = tuple(c for c,b in zip(others118,bits118) if b)
        for activation118 in ("gelu","relu","silu"):
            layouts118.append(score_layout118(vc118,flags118,activation118,sel118,4))
best118 = sorted(layouts118,key=lambda d:d["objective"])[:12]
full118 = []
all118 = np.arange(len(ix104))
for d118 in best118:
    full118.append(score_layout118(d118["value_col"],d118["flags"],
                                  d118["activation"],all118,query112.shape[1]))
full118.sort(key=lambda d:d["objective"])
for d118 in full118[:8]:
    print(d118)
print("Edits:",edited())
with open("/work/notes.md","a") as f:
    f.write("\n## Run 118: joint register-layout search\n"
            "- Tested combined value/presence layouts for the actual eight-input last_numeric projection.\n"
            "- Best full-probe result: "+str(full118[0])+"\n"
            "- Preprocessing assay only; no weights edited. Python runs: 118; forward passes: 4.\n")
```