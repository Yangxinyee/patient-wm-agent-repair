The decoder inspection confirms that value predictions depend on the prior state and the same-code reading register. I’ll rebuild those inputs for training records and audit them against the existing tuning helper. Only an exact register match will justify fitting a calibration table.

```python
import numpy as np, pandas as pd, torch
from pathlib import Path
torch.set_num_threads(8)

codes162 = {c["name"]:c["id"] for c in VOCAB["codes"]}
numeric162 = {(r["code"],r["attribute"]):r for r in VOCAB["numeric"]}
hm162 = head_meta
reg162 = register8.detach().cpu().numpy()
print("REGISTER COLUMNS:", [
    {"col":j,"min":float(reg162[:,j].min()),"max":float(reg162[:,j].max()),
     "rms":float(np.sqrt(np.mean(reg162[:,j]**2)))}
    for j in range(8)])
print("NONEMPTY REGISTER EXAMPLES:")
ix162 = np.flatnonzero(hm162.has_last.to_numpy() > 0)[:3]
print(hm162.iloc[ix162][["code","last_z","last_bin","age","has_last"]].to_dict("records"))
print(reg162[ix162].tolist())

base162 = (W0["embed.code.weight"][torch.tensor(hm162.code.to_numpy())]
           + linear0(register8,"last_numeric")
           + W0["embed.bins.value.weight"][torch.tensor(hm162.last_bin.to_numpy())]
           + W0["embed.bins.rate.weight"][0])
print("STATIC CHECKS:", {
    str(status):float(torch.max(torch.abs(
        base162 + (0 if status is None else W0["last_status.weight"][status]) - static_all122)))
    for status in [None,0,1]})

def reading_metadata162(records, archive, bin_side="right"):
    rows = []
    sx = archive["stay_id"]; sh = archive["hour"]
    for sid, group in records.groupby("stay_id",sort=False):
        indices = np.flatnonzero(sx == sid)
        order = np.argsort(sh[indices],kind="stable")
        indices = indices[order]; hours = sh[indices]
        last = {}
        for row in group.sort_values("hour",kind="stable").itertuples(index=False):
            if (row.event,"value") not in numeric162 or pd.isna(row.value):
                continue
            j = np.searchsorted(hours,row.hour,side="left") - 1
            if j < 0:
                continue
            code = codes162[row.event]
            spec = numeric162[(row.event,"value")]
            edges = np.asarray(spec["fine_bin_edges"])
            prev = last.get(code)
            has = float(prev is not None)
            age = (row.hour-prev[0])*3600 if prev is not None else 0.
            zlast = (prev[1]-spec["median"])/spec["iqr"] if prev is not None else 0.
            lastbin = int(np.searchsorted(edges,prev[1],side=bin_side)+1) if prev is not None else 0
            targetbin = int(np.searchsorted(edges,row.value,side=bin_side)+1)
            rows.append((int(sid),float(row.hour),row.event,code,int(indices[j]),
                         float((row.hour-hours[j])*3600),zlast,lastbin,age,has,
                         float(row.value),targetbin))
            last[code] = (row.hour,row.value)
    return pd.DataFrame(rows,columns=[
        "stay_id","hour","event","code","state_row","dt","last_z","last_bin",
        "age","has_last","observed","target_bin"])

tune_records162 = pd.read_csv("/data/tuning.csv",
    usecols=["stay_id","hour","event","value","rate"])
tune_records162 = tune_records162[tune_records162.stay_id.isin(range(2020000000,2020000100))]
tune_archive162 = np.load("/work/fwd_3/states_original.npz")
for side162 in ["left","right"]:
    tm162 = reading_metadata162(tune_records162,tune_archive162,side162)
    audit162 = tm162.merge(hm162,on=["stay_id","hour","event"],suffixes=("_new","_old"),
                         validate="one_to_one")
    print("METADATA AUDIT",side162,{
        c:float(np.max(np.abs(audit162[c+"_new"]-audit162[c+"_old"])))
        for c in ["state_row","dt","last_z","last_bin","age","has_last"]})
train_meta162 = reading_metadata162(train_records161,s161,"right")
print("TRAINING READINGS:",train_meta162.shape)
print(train_meta162.groupby("event").size().to_dict())
with open("/work/notes.md","a") as f:
    f.write("\n## Run 162: register and chronological alignment audit\n"
            "Rebuilt training reading metadata; checked prior-state selection and same-code registers "
            "against existing tuning metadata before any calibration. No weights changed. "
            "Python runs: 162; forwards: 22.\n")
```