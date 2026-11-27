The first surrogates explain factual means reasonably well but fail to explain MAP changes under the edit. I won’t use those MAP sensitivities. I’ll add the paired state-change evidence directly to the fit and check it on held-out stays; strong agreement there would make the local response estimates more credible.

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.metrics import mean_absolute_error, r2_score
import inspect

models145, diagnostics145 = {}, []
for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    ids = np.flatnonzero(F143.event.to_numpy() == event)
    tr, va = train144[ids], ~train144[ids]
    lv0 = F143.last_value.to_numpy()[ids]
    missing = ~np.isfinite(lv0)
    median = np.nanmedian(lv0)
    lv = np.where(missing, median, lv0)
    scale = max(np.std(lv), 1)
    lastz = (lv - median) / scale
    spline = SplineTransformer(n_knots=7, degree=3, include_bias=False).fit(lv[:, None])
    age = np.log1p(np.maximum(
        F143.hour.to_numpy()[ids] - F143.last_hour.fillna(F143.hour).to_numpy()[ids], 0))
    base = np.c_[spline.transform(lv[:, None]), missing, age, age**2]
    y = F143.predicted_mean_o.to_numpy()[ids]
    ye = F143.predicted_mean_e.to_numpy()[ids]
    dy = ye - y
    offset = y[tr].mean()
    best = None
    for variant in ["out", "context", "query"]:
        so, se = V144o[variant][ids], V144e[variant][ids]
        xo = np.c_[base, so, so * lastz[:, None]]
        xe = np.c_[base, se, se * lastz[:, None]]
        scaler = StandardScaler().fit(xo[tr])
        zo, ze = scaler.transform(xo), scaler.transform(xe)
        ao = np.c_[zo, np.full(len(ids), 10.)]
        ae = np.c_[ze, np.full(len(ids), 10.)]
        ad = ae - ao
        for pair_weight in [3, 10, 30]:
            X = np.r_[ao[tr], ae[tr], pair_weight * ad[tr]]
            Y = np.r_[y[tr] - offset, ye[tr] - offset, pair_weight * dy[tr]]
            for alpha in [10, 100, 1000]:
                model = Ridge(alpha=alpha, fit_intercept=False).fit(X, Y)
                pred = model.predict(ao[va]) + offset
                dp = model.predict(ad[va])
                mae = mean_absolute_error(y[va], pred)
                dmae = mean_absolute_error(dy[va], dp)
                row = dict(event=event, variant=variant, pair_weight=pair_weight,
                           alpha=alpha, mae=mae, delta_mae=dmae,
                           delta_r2=r2_score(dy[va], dp))
                diagnostics145.append(row)
                score = mae + 4 * dmae
                if best is None or score < best["score"]:
                    best = dict(score=score, diagnostic=row, model=model,
                                scaler=scaler, spline=spline, median=median,
                                scale=scale, offset=offset, variant=variant,
                                ids=ids, base=base, ao=ao, ae=ae)
    models145[event] = best
    print(best["diagnostic"])

print("\n=== VOCABULARY CONVENTIONS ===")
print("numeric_note:", VOCAB.get("numeric_note"))
print("event_kinds:", VOCAB.get("event_kinds"))
print("atom_kinds:", VOCAB.get("atom_kinds"))
print("numeric type:", type(VOCAB.get("numeric")).__name__)
num145 = VOCAB.get("numeric")
if isinstance(num145, dict):
    print("numeric keys:", list(num145)[:15])
    for k in list(num145)[:1]:
        print(k, str(num145[k])[:1500])
print("\n=== LAST-NUMERIC ASSAY CONVENTION ===")
try:
    print(inspect.getsource(score_layout118)[:1600])
except Exception:
    compact_code95(score_layout118, limit=1600)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 145: paired-response surrogate assay\n"
            "- Original-only readouts failed MAP edit-delta validation; not used for routing.\n"
            "- Added paired official state/prediction changes to ridge fits, retaining stay-held-out validation.\n"
            "- Results retained as models145/diagnostics145; edits unchanged.\n"
            "- Python runs: 145; forward passes: 10.\n")
```