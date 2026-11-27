Action-bundle posterior error is now about 0.0021, under 1% of the latent state’s scale. That supports testing this pathway. I’ll measure which latent perturbations actually lower glucose or raise MAP, then check those directions on separate stays. Reversed or highly variable responses would refute their usefulness.

```python
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import time

tstart153 = time.time()
sensor_names153 = ["glucose_finger","glucose_lab","map_arterial","map_cuff"]

last_tables153 = {}
for (stay,event), g in records[records.event.isin(sensor_names153)].groupby(
        ["stay_id","event"]):
    g = g.sort_values("hour")
    last_tables153[(int(stay),event)] = (
        g.hour.to_numpy(float),g.value.to_numpy(float))

def centres153(event):
    table = VOCAB["value_bin_centres"]
    value = table.get(event, table.get(str(codes[event])))
    if isinstance(value,dict):
        value = value.get("centres",value.get("centers"))
    return torch.tensor(value,dtype=torch.float32)

def readout153(h,out,stay,current_hour,event,type_index=1):
    next_hour = np.floor(current_hour)+1
    gap_seconds = float((next_hour-current_hour)*3600)
    flowed = h*torch.exp(
        -gap_seconds*torch.exp(W["core.flow.a"])[None,None,:])
    ctx = (out + norm136(flowed.mean(1),"core.out_norm")
           + mlp136(torch.full((len(out),1),np.log1p(gap_seconds)/10),"gap"))
    prior_mean = mlp136(ctx,"z_prior")[:,:32]
    view = ctx + linear136(prior_mean,"z_head")
    q = norm136(view+W["type_emb.weight"][type_index],"query_norm")
    register = torch.zeros((len(out),8))
    last_bin = 0
    table = last_tables153.get((int(stay),event))
    if table is not None:
        pos = np.searchsorted(table[0],current_hour,side="right")-1
        if pos >= 0:
            last_hour,last_value = table[0][pos],table[1][pos]
            spec = numeric[(event,"value")]
            register[:,0] = 1
            register[:,3] = 1
            register[:,1] = np.log1p(next_hour-last_hour)/10
            register[:,6] = np.clip(
                (last_value-spec["median"])/max(spec["iqr"],1e-8),-10,10)
            last_bin = 1+np.searchsorted(
                spec["fine_bin_edges"],last_value,side="left")
    with_code = (
        q + W["embed.code.weight"][codes[event]]
        + linear136(register,"last_numeric")
        + W["embed.bins.value.weight"][last_bin]
        + W["embed.bins.rate.weight"][0]
        + W["last_status.weight"][0])
    logits = (mlp136(with_code,"attr_heads.value_bin")
              + W["embed_transition.value_bin.weight"][codes[event]*17+last_bin])
    return torch.softmax(logits,1) @ centres153(event)

eligible153 = active152[(hours[active152]>=2) & (hours[active152]<24)]
rng153 = np.random.default_rng(153)
chosen_stays153 = rng153.choice(np.unique(sid[eligible153]),24,replace=False)
probe153 = np.array([
    rng153.choice(eligible153[sid[eligible153]==stay])
    for stay in chosen_stays153],dtype=int)

eps153 = .05
perturb153 = torch.cat(
    [torch.zeros((1,32)),eps153*torch.eye(32),-eps153*torch.eye(32)],0)
jac153, base153, errors153 = [], [], []
for j in probe153:
    prevj = int(p152[j])
    history = np.flatnonzero((sid==sid[j]) & (hours<=hours[j]))[-128:]
    memory = e152[history][None,:,:].expand(65,-1,-1)
    ages = torch.tensor(
        (hours[j]-hours[history])*3600,dtype=torch.float32
    )[None,:].expand(65,-1)
    h_before = torch.from_numpy(
        S[prevj,192:960].reshape(1,4,192)).expand(65,-1,-1)
    dt = torch.full((65,),float((hours[j]-hours[prevj])*3600))
    z_input = torch.from_numpy(S[j,960:992])[None,:]+perturb153
    h_new,out_new = core_step136(
        h_before,dt,e152[j][None,:].expand(65,-1),
        u152[j][None,:].expand(65,-1),z_input,memory,ages)
    predictions = torch.stack([
        readout153(h_new,out_new,sid[j],hours[j],event)
        for event in sensor_names153],1).numpy()
    jac153.append(((predictions[1:33]-predictions[33:65])/(2*eps153)).T)
    base153.append(predictions[0])
    errors153.append([
        float(torch.sqrt(torch.mean(
            (h_new[0]-torch.from_numpy(S[j,192:960].reshape(4,192)))**2))),
        float(torch.sqrt(torch.mean(
            (out_new[0]-torch.from_numpy(S[j,:192]))**2)))])
jac153 = np.stack(jac153)
base153 = np.stack(base153)
errors153 = np.array(errors153)
mean_j153 = jac153[:12].mean(0)
ridge153 = .01*float(np.linalg.norm(mean_j153,2)**2)
targets153 = np.array([[-1,0],[-1,0],[0,1],[0,1]],float)
directions153 = mean_j153.T @ np.linalg.solve(
    mean_j153@mean_j153.T+ridge153*np.eye(4),targets153)

print("Finite-difference sensor order:",sensor_names153)
print("Mean recurrent errors (h,out):",errors153.mean(0))
print("Mean Jacobian row norms:",np.linalg.norm(mean_j153,axis=1))
print("Jacobian singular values:",np.linalg.svd(mean_j153,compute_uv=False))
print("Latent direction norms:",np.linalg.norm(directions153,axis=0))
print("Training mean responses:\n",mean_j153@directions153)
responses153 = jac153[12:]@directions153
print("Held-stay mean responses:\n",responses153.mean(0))
print("Held-stay response standard deviations:\n",responses153.std(0))
print("Baseline mean predictions:",base153.mean(0))
print("Latent direction coordinates:\n",directions153.round(5))

factual_reference153 = pd.read_csv("/work/fwd_4/factual_original.csv")
print("Factual reference columns:",list(factual_reference153.columns))
print(factual_reference153.head(3).to_string(index=False))
print("Elapsed seconds:",round(time.time()-tstart153,2))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Python 153 — actual response directions\n"
        "- Finite-differenced posterior means through reconstructed recurrence "
        "and shared value head on 24 distinct training stays.\n"
        f"- Direction norms: {np.linalg.norm(directions153,axis=0).tolist()}.\n"
        f"- Held-stay response means: {responses153.mean(0).tolist()}.\n"
        "- Approximate CPU response calibration only; no weights changed. "
        "Any candidate requires paired GPU rollouts and factual validation.\n"
    )
```
