Glucose calibration did not generalize to held-out training stays, so I’m rejecting it despite its tuning improvement. MAP calibration passed both checks. I’ll combine only that calibration with several existing MAP directions and compare their local responses at a fixed loss budget; this will identify a safer linear fallback.

```python
def scalar_mean192(mc, base_logits, coef, centres, mask):
    mask = np.asarray(mask, dtype=bool)
    cc = torch.as_tensor(coef, dtype=torch.float32)
    tau = mc["inputs"][:, mask, :] @ cc[:-1] + cc[-1]
    pp = (base_logits[:, mask, :] + tau[:, :, None]*wm_v151).softmax(-1)
    ctr = torch.as_tensor(centres, dtype=torch.float32)[mask]
    return (pp.mean(0)*ctr).sum(-1).numpy()

map192_cal = cal190_candidates[(0.1, 10.0)]
map192_centres = wm_train145["centres"]
map192_nebase = {
    factor: head177_eval(wm_cf146[("norepinephrine",factor)],
                         head177_weights, centres=map192_centres)
    for factor in [0.5,2.0]
}
map192_candidates = {}
map192_rows = []
configs192 = [(1.0,1.5)] + [
    (fw,gain) for fw in [0.3,1.0,3.0,10.0]
    for gain in [1.0,2.0,3.0,4.0,6.0,8.0]
]
print("FIT VALUE TYPE:", type(map164_fits[(0.0,1.0,0.001)]).__name__)
for fw192, gain192 in configs192:
    source192 = np.asarray(map164_fits[(0.0,fw192,0.001)], dtype=np.float64)
    assert source192.shape == (193,)
    coef192 = gain192*source192 + map192_cal
    delta192 = coef192-head177_coef
    map192_candidates[(fw192,gain192)] = coef192
    nt192 = cal190_base_train["nll"].copy()
    nu192 = cal190_base_tune["nll"].copy()
    nt192[map172_MAP_train] = scalar_nll190(
        wm_train145, cal190_base_train["logits"], delta192, map172_MAP_train
    )
    nu192[map172_MAP_tune] = scalar_nll190(
        wm_tune145, cal190_base_tune["logits"], delta192, map172_MAP_tune
    )
    factual_mean192 = cal190_base_train["mean"].copy()
    factual_mean192[map172_MAP_train] = scalar_mean192(
        wm_train145, cal190_base_train["logits"], delta192,
        map192_centres, map172_MAP_train
    )
    row192 = {
        "fw":fw192,"gain":gain192,
        "cv_total_dNLL":np.mean((nt192-orig190_train)[~wm_split151]),
        "tune_total_dNLL":np.mean(nu192-orig190_tune),
        "norm":np.linalg.norm(coef192)
    }
    for factor192 in [0.5,2.0]:
        cf192 = wm_cf146[("norepinephrine",factor192)]
        cfmean192 = map192_nebase[factor192]["mean"].copy()
        cfmean192[map172_MAP_train] = scalar_mean192(
            cf192,map192_nebase[factor192]["logits"],delta192,
            map192_centres,map172_MAP_train
        )
        for event192, short192 in [("map_arterial","A"),("map_cuff","C")]:
            sel192 = (rebuilt_train145.event.to_numpy()==event192) & np.asarray(cf192["active"],dtype=bool)
            dr192 = np.asarray(cf192["delta_rate"])[sel192]
            shift192 = (cfmean192-factual_mean192)[sel192]
            row192[f"{short192}{factor192}_slope"] = np.dot(dr192,shift192)/np.dot(dr192,dr192)
    map192_rows.append(row192)

map192_table = pd.DataFrame(map192_rows)
safe192 = map192_table[
    (map192_table.cv_total_dNLL < 0.0047) &
    (map192_table.tune_total_dNLL < 0.0047)
].sort_values("A2.0_slope",ascending=False)
print("\nCANDIDATES WITH LOCAL FACTUAL MARGIN")
print(safe192.to_string(index=False))
print("\nNEXT FOUR OUTSIDE THE MARGIN")
print(map192_table[
    ~map192_table.index.isin(safe192.index)
].sort_values("tune_total_dNLL").head(4).to_string(index=False))
map192_table.to_csv("/work/map_calibrated_frontier.csv",index=False)

print("\nINSULIN COMPRESSION CHECK")
for j192, sign192 in [(170,1),(89,-1)]:
    print(j192, "max output-column deviation:",
          float((head177_weights[head177_names[2]][:,j192]-sign192*wm_v151).abs().max()))

with open("/work/notes.md","a") as f:
    f.write("\n## Run 192: calibrated linear MAP frontier\n"
            "- Rejected glucose calibration: held-out training loss worsened at every tested regularisation level.\n"
            "- Combined protected MAP calibration (ridge .1, contrast penalty 10) with existing raw-input directions.\n"
            "- No new weights installed; screening uses held-out training and tuning factual losses.\n"
            + safe192.to_string(index=False) + "\n")
```
