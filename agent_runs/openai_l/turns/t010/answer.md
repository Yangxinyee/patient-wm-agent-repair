Insulin dosing jumps at several glucose thresholds. Those jumps may help distinguish drug effects from smooth changes in patient condition. I’ll estimate the next-reading discontinuities and check a second, pre-dose sensor as a placebo. A post-dose jump without a pre-dose jump would support causality; jumps in both would undermine it.

```python
import numpy as np, pandas as pd
from scipy import stats

print("=== TARGETED MODEL DETAILS ===")
print("VOCAB keys:", list(VOCAB))
for k,v in VOCAB.items():
    if k in ["codes","config"] or any(s in k.lower() for s in ["normal","scale"]):
        print(k, str(v)[:1600])
for n in NAMES:
    if n.startswith(("embed.numeric","embed.bins.","encode_action.","attr_heads.value_bin","last_numeric","query_norm")):
        print(n, SHAPES[n])
print("Kalman names:", kalman_nll.__code__.co_varnames)
print("Kalman constants:", str(kalman_nll.__code__.co_consts)[:1300])
print("Rich fits: drug coefficient and reported SE only")
for key in rich_fits:
    if "both sensors current" in key:
        v = rich_fits[key]
        j = 1 if key[0].startswith("glucose") else 2
        print(key, "beta", round(float(v[0][j]),4), "SE", round(float(v[1][j]),4))

# Canonical last readings, including the pre-ICU readings at hour -1.
pre = train.loc[train.event.isin(P) & train.hour.eq(-1)].pivot(
    index="stay_id", columns="event", values="value").reindex(rec_ids)
Last = {}
for e in P:
    mat = np.column_stack([pre[e].to_numpy(), P[e][:,:25]])
    pos = np.maximum.accumulate(np.where(np.isfinite(mat), np.arange(26), 0), axis=1)
    Last[e] = np.take_along_axis(mat, pos, axis=1)[:,1:]

gid_rd = np.repeat(np.arange(len(rec_ids)),24)
hour_rd = np.tile(np.arange(24),len(rec_ids))

def discontinuity(x, y, a, cutoff, width):
    x,y,a = np.asarray(x).ravel(),np.asarray(y).ravel(),np.asarray(a).ravel()
    sel = np.isfinite(x)&np.isfinite(y)&np.isfinite(a)&(np.abs(x-cutoff)<width)
    r = x[sel]-cutoff
    side = (r>0).astype(float)
    w = 1-np.abs(r)/width
    h = hour_rd[sel]
    X = np.column_stack([np.ones(len(r)),side,r,side*r,
                         h/24,(h/24)**2])
    Xw = X*w[:,None]
    inv = np.linalg.pinv(X.T@Xw)
    by = inv@(Xw.T@y[sel])
    ba = inv@(Xw.T@a[sel])
    beta = by[1]/ba[1]
    influence = (Xw@inv[:,1]) * ((y[sel]-X@by)-beta*(a[sel]-X@ba))
    cluster = np.bincount(gid_rd[sel],weights=influence,minlength=len(rec_ids))
    se = np.sqrt(cluster@cluster)/abs(ba[1])
    return len(r),float(ba[1]),float(by[1]),float(beta),float(se)

rd_results = []
fg = P["glucose_finger"][:,:24]
dose = A["insulin"][:,:24]
print("\n=== INSULIN THRESHOLD ESTIMATES: n, dose jump, outcome jump, ratio, clustered SE ===")
for cutoff in [150.5,179.5,219.5,259.5]:
    for width in [6,12]:
        for target in ["glucose_finger","glucose_lab"]:
            ans = discontinuity(fg,P[target][:,1:25],dose,cutoff,width)
            rd_results.append(("insulin",cutoff,width,target,*ans))
            print(cutoff,width,target,tuple(round(z,4) if isinstance(z,float) else z for z in ans))
        ans = discontinuity(fg,P["glucose_lab"][:,:24],dose,cutoff,width)
        print(cutoff,width,"PRE-dose lab placebo",
              tuple(round(z,4) if isinstance(z,float) else z for z in ans))
    ans = discontinuity(fg,np.isfinite(P["glucose_finger"][:,1:25]).astype(float),dose,cutoff,12)
    print(cutoff,"next finger OBSERVATION jump:",round(ans[2],4))

print("\n=== NOREPINEPHRINE POLICY DIAGNOSTICS ===")
a = A["norepinephrine"][:,:24]
ar,cu = P["map_arterial"][:,:24],P["map_cuff"][:,:24]
both_now = np.isfinite(ar)&np.isfinite(cu)
for label,mask in [
    ("both >=65",both_now&(ar>=65)&(cu>=65)),
    ("arterial <65, cuff >=65",both_now&(ar<65)&(cu>=65)),
    ("arterial >=65, cuff <65",both_now&(ar>=65)&(cu<65)),
    ("both <65",both_now&(ar<65)&(cu<65))]:
    print(label,"n",int(mask.sum()),"active",round(np.mean(a[mask]>0),4),"mean rate",round(np.mean(a[mask]),4))
for source in ["map_arterial","map_cuff"]:
    q = pd.DataFrame({"reading":P[source][:,:24].ravel(),"rate":a.ravel(),
                      "both":both_now.ravel()})
    tab = q.loc[q.both & q.reading.between(59,72)].groupby("reading").agg(
        n=("rate","size"),active=("rate",lambda z:np.mean(z>0)),mean=("rate","mean"))
    print(source,tab.round(4).to_string())

with open("/work/notes.md","a") as f:
    f.write("\n## Run 8\nInsulin has sharp recorded-glucose dosing thresholds. Computing local-linear discontinuity ratios at 150.5,179.5,219.5,259.5, two bandwidths, both next-hour sensors, and pre-dose lab/observation-placebo checks. Also testing which MAP sensor drives NE selection. Results remain subject to placebo and precision checks.\n")
```