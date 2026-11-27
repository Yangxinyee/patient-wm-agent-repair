Query type 1 is the best match, and the base attributes contain no numeric register fields. The prior test therefore omitted “has last” and age as well as value information. I’ll add those fields and test their ordering, while checking whether prior sampling explains any remaining error. A sharp improvement would confirm the register diagnosis.

```python
import pandas as pd
tune121 = pd.read_csv("/data/tuning.csv")
name_by_code121 = {x["id"]:x["name"] for x in VOCAB["codes"]}
selected121 = []
for c, target in zip(codes97, observed_mean97):
    sub = F97[F97.event == name_by_code121[int(c)]]
    j = np.argmin(np.abs(sub.predicted_mean.to_numpy()-float(target)))
    selected121.append(sub.iloc[j])
readrows121 = pd.DataFrame(selected121).reset_index(drop=True)

age121, lastv121 = [], []
groups121 = {
    key: g.sort_values("hour")
    for key,g in tune121[tune121.event.isin(
        ["glucose_finger","glucose_lab","map_arterial","map_cuff"]
    )].groupby(["stay_id","event"])
}
for row in readrows121.itertuples(index=False):
    previous = groups121[(row.stay_id,row.event)]
    previous = previous[previous.hour < row.hour]
    age121.append(float(row.hour-previous.iloc[-1].hour))
    lastv121.append(float(previous.iloc[-1].value))
age121 = np.array(age121)

flow121 = torch.as_tensor(x97[:,192:960].reshape(-1,4,192)) * torch.exp(
    -torch.as_tensor(dt97,dtype=torch.float32)[:,None,None]
    * torch.exp(W["core.flow.a"])
)
ctx121 = torch.as_tensor(x97[:,:192]) + layer_norm(flow121.mean(1),"core.out_norm")
ctx121 += mlp(torch.log(torch.as_tensor(dt97,dtype=torch.float32))[:,None]/10,"gap")
prior121 = mlp(ctx121,"z_prior")
view121 = ctx121 + linear(prior121[:,:32],"z_head")
print("context versus retained RMS",float(torch.sqrt(torch.mean((ctx121-ctx97)**2))))
print("view versus retained RMS",float(torch.sqrt(torch.mean((view121-view97)**2))))
print("prior SD quantiles",
      np.quantile(torch.exp(prior121[:,32:]).numpy(),[0,.5,.9,1]))

query121 = layer_norm(view121+W["type_emb.weight"][1],"query_norm")
age_choices121 = {
    "log1p_seconds/10":np.log1p(age121*3600)/10,
    "log_seconds/10":np.log(age121*3600)/10,
    "log1p_hours/10":np.log1p(age121)/10,
    "hours/10":age121/10,
    "zero":np.zeros_like(age121)
}
candidate_meta121, attrs121 = [], []
for hc,ac in ((0,1),(1,0)):
    for agename,agefeature in age_choices121.items():
        regbase = query121+base120+W["last_numeric.weight"][:,hc][None]
        regbase += torch.tensor(agefeature,dtype=torch.float32)[:,None] * W["last_numeric.weight"][:,ac][None]
        for pc in range(2,8):
            for vc in range(2,8):
                if pc == vc:
                    continue
                attrs121.append(regbase + W["last_numeric.weight"][:,pc][None]
                                + zc120*W["last_numeric.weight"][:,vc][None])
                candidate_meta121.append((hc,ac,agename,pc,vc))
logits121 = mlp(torch.stack(attrs121),"attr_heads.value_bin")+trans120[None]
predictions121 = (torch.softmax(logits121,-1)*cent120[None]).sum(-1).numpy()
errors121 = np.sqrt(np.mean((predictions121-truth120[None])**2,axis=1))
best_indices121 = np.argsort(errors121)[:10]
print("=== COMPLETE REGISTER CANDIDATES ===")
for j in best_indices121:
    print(float(errors121[j]),candidate_meta121[j])
jbest121 = int(best_indices121[0])
attrbest121 = attrs121[jbest121] - query121
print("best per-code RMSE",
      {int(c):float(np.sqrt(np.mean((predictions121[jbest121][codes97==c]
       -truth120[codes97==c])**2))) for c in np.unique(codes97)})
```
