This repeats the baseline rollout, so it adds no new evidence about the drugs’ true effects. I still need the stronger record-based checks before choosing an edit. I’ll run—or recover, if already computed—the nonlinear and sensor-restricted regressions. Consistency across these tests would strengthen the case for a real response.

```python
RUNS += 1
FORWARDS = 1
from threadpoolctl import threadpool_limits
threadpool_limits(limits=8)
torch.set_num_threads(8)

if "Xrich" not in globals():
    more, more_names = [], []
    for e in events[:4]:
        x = FF[e][:,1:25]
        knots = ([100,130,160,190,220,250] if "glucose" in e
                 else [45,55,65,75,85,95])
        for knot in knots:
            more.append(np.maximum(x-knot,0).ravel())
            more_names.append(f"{e}_hinge{knot}")
        obs = np.isfinite(A[e])
        lastpos = np.maximum.accumulate(np.where(obs, np.arange(26), -1), axis=1)
        more.append((np.arange(26)[None,:]-lastpos)[:,1:25].ravel())
        more_names.append(f"{e}_age")
        count = np.cumsum(obs,axis=1)
        cm = np.cumsum(np.nan_to_num(A[e]),axis=1)/np.maximum(count,1)
        cm = np.where(count>0,cm,np.nanmedian(A[e]))
        more.append(cm[:,1:25].ravel())
        more_names.append(f"{e}_pastmean")
        more.append(np.repeat(FF[e][:,1,None],24,axis=1).ravel())
        more_names.append(f"{e}_admission")
    for e in events[4:]:
        for lag in [2,3]:
            more.append(np.pad(A[e],((0,0),(lag,0)))[:,1:25].ravel())
            more_names.append(f"{e}_lag{lag}")
    Xrich = np.column_stack([Xhistory]+more)
    rich_names = basic_names+history_names+more_names

if "rich_fits" not in globals():
    rich_fits = {}
    policy_fits = {}
    for e,drug,other in [
        ("glucose_finger","insulin","glucose_lab"),
        ("glucose_lab","insulin","glucose_finger"),
        ("map_arterial","norepinephrine","map_cuff"),
        ("map_cuff","norepinephrine","map_arterial")
    ]:
        yn = A[e][:,2:26].ravel()
        active = A[drug][:,1:25].ravel()>0
        own = np.isfinite(A[e][:,1:25]).ravel()
        both = own & np.isfinite(A[other][:,1:25]).ravel()
        j = 1 if drug=="insulin" else 2
        for label,mask in [("all active",active),
                           ("current own observed",active&own),
                           ("both sensors current",active&both)]:
            ok = mask & np.isfinite(yn)
            for within in [False,True]:
                rich_fits[(e,label,within)] = ols_cluster(
                    Xrich[ok],yn[ok],GID[ok],within)
        ok = active & both
        policy_fits[drug] = ols_cluster(
            np.delete(Xrich[ok],j,axis=1),
            A[drug][:,1:25].ravel()[ok],GID[ok],True)

for (e,label,within),(b,s,sd,n) in rich_fits.items():
    j = 1 if "glucose" in e else 2
    print(e, label, "FE" if within else "pooled",
          f"effect={b[j]:+.4f} SE={s[j]:.4f} SD={sd:.3f} n={n}")
if "policy_fits" in globals():
    for drug,(b,s,sd,n) in policy_fits.items():
        j = 1 if drug=="insulin" else 2
        pn = rich_names[:j]+rich_names[j+1:]
        channels = events[:2] if j==1 else events[2:4]
        print("Policy",drug,"residual SD",round(sd,4),
              "sensor slopes",
              {e:round(b[pn.index(e+"_lag0")],5) for e in channels})
print("\nBaseline rollout file:")
baseline_rollout = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
print(baseline_rollout.columns.tolist())
print(baseline_rollout.head(3).to_string(index=False))
print("Counts:",RUNS,"Python;",FORWARDS,"forward.")
with open("/work/notes.md","a") as f:
    f.write("\n## Baseline rollout\nFirst100 tuning original shifts: "
            "I x2 -3.680, I half +2.018, NE x2 +0.465, NE half -0.871. "
            "Rich regression results retained in rich_fits; "
            "they are diagnostic, not yet accepted causal estimates.\n")
```