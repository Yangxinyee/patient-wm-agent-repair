The calibration run stopped before fitting anything because the tuning-loss helper also requires observed bins. No weights changed. I’ll recover the matching cached bins and rerun the same closed-form calculation; only validation improvement will justify staging a calibration.

```python
expected_tune_delta280 = float(grid269[(.25,.25)]["score"][2])
bin_candidates280 = {}
seen280 = set()

def consider_bins280(label, value):
    if (isinstance(value, np.ndarray) and value.shape == (4751,)
        and value.dtype.kind in "iu" and value.min() >= 0 and value.max() <= 16
        and id(value) not in seen280):
        seen280.add(id(value))
        bin_candidates280[label] = value

for name280, value280 in list(globals().items()):
    consider_bins280(name280, value280)
    if isinstance(value280, dict):
        for key280, subvalue280 in list(value280.items()):
            consider_bins280(name280 + "[" + repr(key280) + "]", subvalue280)

matches280 = []
for name280, b280 in bin_candidates280.items():
    nv280 = np.asarray(nll228(logits_tune279, b280))
    delta280 = float((nv280-base_nll_tune234).mean())
    matches280.append((abs(delta280-expected_tune_delta280), name280, delta280))
print("Bin-cache matches", sorted(matches280)[:6])
assert matches280, "No cached tuning observed-bin vector found."
best280 = min(matches280)
assert best280[0] < 1e-5, "Cached bins do not reproduce the known factual score."
bins280 = bin_candidates280[best280[1]]
ntune279 = np.asarray(nll228(logits_tune279, bins280))

print("Baseline deltas",
      float((ntrain279-base_nll_train234)[fitmask179].mean()),
      float((ntrain279-base_nll_train234)[valmask179].mean()),
      float((ntune279-base_nll_tune234).mean()))
print("Final-head reconstruction error",
      float(np.max(np.abs(Tcurrent279-grid269[(.25,.25)]["T"]))))

events279 = np.asarray(events179)
codes279 = {}
for e279 in np.unique(events279):
    v279 = code_lookup254[e279]
    if isinstance(v279, dict):
        v279 = next(v279[k] for k in ("id", "index", "idx", "code_id") if k in v279)
    codes279[e279] = int(v279)
print("Code IDs", codes279)
tune_codes279 = ix_tune253 // 17
probs_train279 = softmax179(logits_train279, axis=-1).mean(axis=0)
print("Observed-bin agreement", np.array_equal(B179, observed_bins82))

calgrid279 = {}
for alpha279 in (25., 100., 400., 1600.):
    offsets279 = {}
    for e279 in codes279:
        use279 = fitmask179 & (events279 == e279)
        expected279 = probs_train279[use279].sum(axis=0)
        observed279 = np.bincount(B179[use279], minlength=17).astype(float)
        pseudo279 = alpha279 * expected279 / max(int(use279.sum()), 1)
        d279 = np.log((observed279 + pseudo279 + 1e-8) /
                      (expected279 + pseudo279 + 1e-8))
        d279[0] = 0.
        d279[1:] -= np.average(d279[1:], weights=expected279[1:]+1e-8)
        offsets279[e279] = d279
    for strength279 in (.2, .5, 1.):
        dtrain279 = np.zeros((len(events279), 17))
        dtune279 = np.zeros((len(tune_codes279), 17))
        for e279, code279 in codes279.items():
            dtrain279[events279 == e279] = strength279 * offsets279[e279]
            dtune279[tune_codes279 == code279] = strength279 * offsets279[e279]
        nc279 = np.asarray(nll179(logits_train279 + dtrain279[None]))
        nv279 = np.asarray(nll228(logits_tune279 + dtune279[None], bins280))
        score279 = (
            float((nc279-base_nll_train234)[fitmask179].mean()),
            float((nc279-base_nll_train234)[valmask179].mean()),
            float((nv279-base_nll_tune234).mean())
        )
        calgrid279[(alpha279,strength279)] = {
            "offsets": {e: strength279*d for e,d in offsets279.items()},
            "score": score279,
            "relative": (float((nc279-ntrain279)[valmask179].mean()),
                         float((nv279-ntune279).mean()))
        }

print("=== CALIBRATION: deltas from original; gains against current ===")
for key279, item279 in sorted(
    calgrid279.items(), key=lambda kv: max(kv[1]["relative"])
):
    print(key279, np.round(item279["score"], 6),
          "relative", np.round(item279["relative"], 6))

with open("/work/notes.md", "a") as f280:
    f280.write("\n## Runs 279–280 / records-fitted bin calibration\n"
               "Run279 stopped before fitting because nll228 requires bins. "
               "Run280 matched the cached observed-bin vector by reproducing "
               "the previously validated offline score, then fitted "
               "closed-form sensor-specific count offsets on train fitmask only. "
               "No weights saved; candidate scores retained in calgrid279.\n")
```