The new insulin estimate is about −1.8 per rate unit after prior-history adjustment; norepinephrine’s sensor instrument remains weak. The adjusted placebo tests were uninformative because those readings were included as controls. I’ll use earlier placebos and estimate persistence, which determines how one-hour effects accumulate in the rollout.

```python
from scipy.optimize import least_squares

print("=== BASELINE FORWARD SUMMARIES ===")
for folder in ["/work/fwd_1", "/work/fwd_2"]:
    print(folder, json.loads(Path(folder + "/summary.json").read_text()))

print("\n=== EARLIER PLACEBOS NOT INCLUDED IN THE CONTROLS ===")
for k in [-3, -4, -6]:
    print(k, contrast_iv("insulin", "glucose_finger", "glucose_lab",
                        "glucose_finger", k, True))

# Dynamic moments: difference equations eliminate a constant stay intercept.
def persistence_given_effect(outcome, drug, bval, instrument_lag):
    y = panel[outcome]
    dy_next = (pg[outcome].shift(-1)-y).to_numpy()
    dy_now = (y-pg[outcome].shift(1)).to_numpy()
    du = (panel[drug]-pg[drug].shift(1)).to_numpy()
    z = pg[outcome].shift(instrument_lag).to_numpy()
    good = (np.isfinite(dy_next) & np.isfinite(dy_now) &
            np.isfinite(du) & np.isfinite(z) & (hours >= 1) & (hours <= 23))
    C = np.column_stack([np.ones(good.sum()), hours[good]/24])
    V = np.column_stack([z[good], dy_now[good], du[good], dy_next[good]])
    R = V-C@sla.lstsq(C, V, lapack_driver="gelsy")[0]
    zr, dxr, dur, dyr = R.T
    den = zr@dxr
    aval = (zr@(dyr-bval*dur))/den
    _, inv = np.unique(sid[good], return_inverse=True)
    gs = np.bincount(inv, weights=zr*(dyr-aval*dxr-bval*dur))
    se = np.sqrt(gs@gs)/abs(den)
    return float(aval), float(se), int(good.sum())

fixed_b_dynamics = {}
print("\n=== PERSISTENCE CONDITIONAL ON A FIXED ONE-HOUR EFFECT ===")
for outcome, drug, bs in [
    ("glucose_finger", "insulin", [-1.8, -2.4]),
    ("map_arterial", "norepinephrine", [0.0, 0.5, 2.5])
]:
    for bv in bs:
        vals = [persistence_given_effect(outcome, drug, bv, lag) for lag in [2,3,4,5]]
        fixed_b_dynamics[(drug,bv)] = vals
        print(drug, "b=", bv, "a, SE, n for lags 2..5:", np.round(vals,4).tolist())

# Insulin impulse-response moments account for subsequent recorded dose changes.
unique_sid, sid_inv = np.unique(sid, return_inverse=True)
lp_moments = []
for h in range(1,9):
    z0 = (panel.glucose_finger-panel.glucose_lab).to_numpy()
    yh = pg.glucose_finger.shift(-h).to_numpy()
    good = np.isfinite(z0) & np.isfinite(yh) & (hours>=2) & (hours<=24-h)
    useq = np.column_stack([
        pg.insulin.shift(-j).fillna(0).to_numpy()[good] for j in range(h)
    ])
    V = np.column_stack([z0[good], yh[good], useq])
    C = preC[good]
    R = V-C@sla.lstsq(C,V,lapack_driver="gelsy")[0]
    zr = R[:,0]
    # Each column stores a stay's contribution to a covariance moment.
    G = np.column_stack([
        np.bincount(sid_inv[good], weights=zr*R[:,j], minlength=len(unique_sid))
        for j in range(1,R.shape[1])
    ]) / good.sum()
    lp_moments.append(G)

def lp_scores(theta):
    aval,bval = theta
    return np.column_stack([
        G[:,0]-bval*(G[:,1:]@(aval**np.arange(h-1,-1,-1)))
        for h,G in enumerate(lp_moments,1)
    ])

scales = np.sqrt((lp_scores([0.8,-2.0])**2).sum(axis=0))
lp_fit = least_squares(lambda th: lp_scores(th).sum(axis=0)/scales,
                       [0.8,-2.0], bounds=([0.0,-15.0],[1.1,5.0]))
print("\n=== INSULIN STRUCTURAL IMPULSE-RESPONSE FIT ===")
print("a,b:", lp_fit.x, "standardised moment residuals:", lp_fit.fun)
print("raw horizon-specific IV ratios:",
      [round(G[:,0].sum()/G[:,1].sum(),3) for G in lp_moments])

# Convert candidate dynamics into the evaluation's fixed-dose contrasts.
def linear_readouts(aval,bval,drug,outcome,df=tpanel):
    result = {}
    ds = df[drug].unstack(level=-1).fillna(0)
    eligible = ds.loc[:,12:23].gt(0).cumsum(axis=1).gt(0)
    for factor in [2.0,0.5]:
        delta = np.zeros(len(ds))
        vals = []
        masks = []
        for t in range(12,24):
            delta = aval*delta+bval*(factor-1)*ds[t].to_numpy()
            vals.append(delta.copy())
            masks.append(eligible[t].to_numpy())
        V,M = np.column_stack(vals),np.column_stack(masks)
        per_stay = np.divide((V*M).sum(axis=1),M.sum(axis=1),
                             out=np.full(len(ds),np.nan),where=M.sum(axis=1)>0)
        result[factor] = float(np.nanmean(per_stay))
    return result

print("\n=== CANDIDATE FIXED-DOSE READOUTS, NOT ESTABLISHED TRUTH ===")
for av,bv in [(0.6,-1.8),(0.8,-1.8),(0.9,-1.8),tuple(lp_fit.x)]:
    print("insulin",av,bv,linear_readouts(av,bv,"insulin","glucose_finger"))
for av,bv in [(0.6,0),(0.8,0.5),(0.8,2.5)]:
    print("norepinephrine",av,bv,linear_readouts(av,bv,"norepinephrine","map_arterial"))

with open("/work/notes.md","a") as f:
    f.write("\n## Dynamic identification; Python 10, forward 2\n"
            "- Prior-controlled sensor IV supports a modest negative immediate insulin effect; NE sensor instrument is weak.\n"
            "- Prior-controlled placebo lags 1/2 were tautologies; added earlier placebos.\n"
            "- Estimated persistence conditional on candidate effects and insulin impulse moments including later doses.\n"
            + json.dumps({"fixed_b_dynamics":{str(k):v for k,v in fixed_b_dynamics.items()},
                          "insulin_lp_fit":lp_fit.x.tolist(),
                          "lp_standardised_residuals":lp_fit.fun.tolist()},indent=2)+"\n")
```