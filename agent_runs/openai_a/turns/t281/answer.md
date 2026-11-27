Across all 400 tuning stays, the candidate’s factual NLL increase is about 0.00406, below the 0.005 limit. The main unresolved weakness is insulin halving. I’ll examine the gate algebra, synthetic dose scores, and stay-level uncertainty before changing weights. If reduced doses move the gate toward inactivity, that would support reshaping it; otherwise we need another explanation.

```python
import dis, pandas as pd
frames = []
for k in [27,28,30,31]:
    a = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    b = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    if k == 27:
        print("FACTUAL COLUMNS", list(a.columns))
    nc = next(c for c in a if "nll" in c.lower())
    sid = next(c for c in a if "stay" in c.lower())
    assert len(a) == len(b)
    keycols = [c for c in a if c in b and any(s in c.lower() for s in ["stay","hour","event"])]
    assert a[keycols].equals(b[keycols])
    frames.append(pd.DataFrame({"stay_id":a[sid], "delta":b[nc]-a[nc]}))
gated8_factual_deltas = pd.concat(frames, ignore_index=True)
agg = gated8_factual_deltas.groupby("stay_id").delta.agg(["sum","count"])
rng = np.random.default_rng(240)
draw = rng.integers(0,len(agg),size=(4000,len(agg)))
boot = agg["sum"].to_numpy()[draw].sum(1)/agg["count"].to_numpy()[draw].sum(1)
print("ALL400 NLL DELTA", gated8_factual_deltas.delta.mean(),
      "STAY BOOTSTRAP 95%", np.quantile(boot,[.025,.975]))
with open("/work/notes.md","a") as f:
    f.write(f"\n## Full400 factual validation; Python 241, forwards 31/50\n"
            f"- Gated8 NLL delta {gated8_factual_deltas.delta.mean():.8f}; "
            f"stay-bootstrap 95% interval {np.quantile(boot,[.025,.975]).tolist()}.\n"
            "- Current candidate unchanged. Investigating halving asymmetry.\n")
print("\n=== CONTRAST GRAFT INSTRUCTIONS ===")
for ins in dis.get_instructions(make_contrast_graft):
    print(ins.opname, ins.argrepr)
print("\n=== SELECTIVE GRAFT INSTRUCTIONS ===")
for ins in dis.get_instructions(make_selective_contrast_graft):
    print(ins.opname, ins.argrepr)
print("\n=== SYNTHETIC/CACHE STRUCTURES ===")
print("sc_nodes",sc_nodes, "sc_names",sc_names)
for k,v in sc_syn.items():
    print("sc_syn",k,type(v).__name__,
          {kk:getattr(vv,"shape",type(vv).__name__) for kk,vv in v.items()} if isinstance(v,dict) else getattr(v,"shape",None))
for k,v in sc_fact.items():
    print("sc_fact",k,type(v).__name__,
          {kk:getattr(vv,"shape",type(vv).__name__) for kk,vv in v.items()} if isinstance(v,dict) else getattr(v,"shape",None))
print("activity score quantiles",np.quantile(score_activity,[0,.01,.1,.5,.9,.99,1]))
print("candidate equality",
      {n:float(torch.max(torch.abs(load(n)-validated_gated8_checkpoint[n]))) for n in edited()})
```
