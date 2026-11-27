The architecture sends doses through a separate action encoder, but there is no explicit drug-effect parameter to adjust. Earlier checks also found that insulin thresholds change which finger readings get recorded, so lab readings are the cleaner effect check. I’ll inspect the dose policy and relevant weights next; a credible norepinephrine estimate must survive adjustment for prior condition and checks for recording-policy changes.

```python
print("=== PANEL STRUCTURE ===")
print("index:", panel.index.names, "columns:", panel.columns.tolist())
print(panel.head(10).to_string())
print("\n=== TIMING ===")
print(train.groupby("event")["hour"].agg(["count", "min", "max"]).to_string())
print(train[train.event.isin(["insulin", "norepinephrine"])].head(8).to_string(index=False))

print("\n=== VOCAB CONFIG AND NUMERIC ===")
print(json.dumps(VOCAB["config"], indent=2))
print(json.dumps(VOCAB["numeric"], indent=2))
print("codes:", VOCAB["codes"])

print("\n=== RELEVANT WEIGHT SHAPES ===")
for name in NAMES:
    if any(s in name for s in [
        "embed.numeric", "encode_action", "attr_heads.value", "last_numeric",
        "query_norm", "type_emb", "z_head", "core.z_", "core.prelude",
        "core.out_norm", "embed_transition.value"
    ]):
        print(name, SHAPES[name])

print("\n=== EXISTING FUNCTIONS ===")
print([k for k, v in list(globals().items())
       if callable(v) and getattr(v, "__module__", "") == "__main__"])

print("\n=== NOREPINEPHRINE POLICY BY CURRENT MAP ===")
pp = panel.reset_index()
print(pp[["map_arterial", "map_cuff", "norepinephrine", "insulin"]].describe().to_string())
for sensor in ["map_arterial", "map_cuff"]:
    sub = pp[pp[sensor].notna()].copy()
    sub["map_group"] = (sub[sensor] / 2).round() * 2
    policy = sub.groupby("map_group").agg(
        n=(sensor, "size"),
        mean_NE=("norepinephrine", "mean"),
        active_NE=("norepinephrine", lambda x: (x > 0).mean())
    )
    print(sensor)
    print(policy.loc[50:90].round(4).to_string())

print("\n=== FORWARD FILE HEADERS ===")
for folder in ["/work/fwd_1", "/work/fwd_2"]:
    for f in Path(folder).glob("*.csv"):
        df = pd.read_csv(f)
        print(str(f), df.shape, df.columns.tolist())
        print(df.head(2).to_string(index=False))
```
