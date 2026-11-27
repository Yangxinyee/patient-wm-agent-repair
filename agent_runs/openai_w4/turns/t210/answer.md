The posterior channel is rank one, and the reserved attention head has unused value dimensions. I’ll test the norepinephrine innovation decoders on factual, half-dose and double-dose encodings before adding a second channel. A useful decoder should track the imposed dose change while the gate suppresses untreated bundles.

```python
from scipy.special import expit

def r2_185(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = np.asarray(y)[ok], np.asarray(p)[ok]
    return 1 - np.sum((y-p)**2) / max(np.sum((y-y.mean())**2), 1e-12)

def gated_ne185(raw, coef, scale=40.0):
    raw = as_np184(raw)
    q = raw @ coef[:576] + coef[576]
    gate = gate_fits180[100.0].decision_function(raw[:,384:576]) * scale
    a, b = gate + q/2, gate - q/2
    return a*expit(a) - b*expit(b)

hold185 = np.asarray(dm87, bool) & np.asarray(te87, bool)
inactive185 = ~np.asarray(active_ne180, bool)
delta185 = np.asarray(dr87)
print("TARGET CHECK",
      "held active", int(hold185.sum()),
      "target quantiles", np.nanquantile(y87[hold185], [.05,.5,.95]),
      "double-dose increment quantiles", np.nanquantile(delta185[hold185], [.05,.5,.95]))

for alpha in (.001, .01, .1, 1.0):
    coef = raw_candidates87[("norepinephrine", alpha, 10.0)]
    q = gated_ne185(raw87, coef)
    summaries = []
    for factor in (.5, 2.0):
        qcf = gated_ne185(raw_cf87[("norepinephrine", factor)], coef)
        actual = (factor-1)*delta185
        use = hold185 & np.isfinite(actual) & (abs(actual)>1e-8)
        change = qcf[use]-q[use]
        slope = np.dot(actual[use], change) / np.dot(actual[use], actual[use])
        rmse = np.sqrt(np.mean((change-actual[use])**2))
        summaries.append((factor, round(float(slope),5), round(float(rmse),5)))
    print("DECODER", alpha,
          "held factual R2", round(float(r2_185(y87[hold185],q[hold185])),5),
          "inactive RMS", round(float(np.sqrt(np.mean(q[inactive185]**2))),6),
          "dose slopes / increment RMSE", summaries)

print("\nCHANNEL VECTOR CHECKS")
gamma185 = as_np184(gamma104)
bias185 = as_np184(bias104)
sig185 = as_np184(signature104)
car185 = as_np184(carrier104)
val185 = as_np184(value104)
print("value104 norm", np.linalg.norm(val185),
      "bias dot", val185 @ bias185,
      "signature dot", val185 @ (gamma185*sig185),
      "carrier gain", val185 @ (gamma185*car185),
      "formula error", np.linalg.norm(val185-car185/gamma185/(lambda104*np.sqrt(192))))
print("S104 symmetric error / trace",
      np.linalg.norm(S104-S104.T), np.trace(S104))
print("capture keys",
      [(k,getattr(v,"shape",type(v).__name__)) for k,v in cap184[1].items()])
print("repair bank keys", list(repair_bank))
print("repair records columns", list(repair_records.columns))
print("probe columns", list(probe_data95.columns) if hasattr(probe_data95,"columns") else type(probe_data95))
print("probe event counts", np.unique(probe_events118,return_counts=True))
print("direction norms", [(k,round(float(np.linalg.norm(v)),4)) for k,v in directions118.items()])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 185\n"
        "Verified a rank-one latent channel and unused value dimensions in "
        "the existing reserved attention head. Tested gated NE innovation "
        "decoders on factual and imposed half/double-dose encodings. These "
        "tests concern faithful input transmission, not causal truth. "
        "No weights changed.\n"
    )
```