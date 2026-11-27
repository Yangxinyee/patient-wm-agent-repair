The saved head matches the balanced solution. The reweighted regression uses only doubled and halved norepinephrine doses, which may explain its difference from the earlier multi-dose fit. I’ll verify that explanation and reproduce the candidate exactly; if both checks pass, I’ll stage it for GPU validation.

```python
print("=== EXPLAIN BASELINE DISCREPANCY ===")
moment_errors285 = {}
for drug in ["insulin", "norepinephrine"]:
    factors = [0.0, 0.5, 1.5, 2.0, 3.0]
    Gall = np.mean([partial281[(drug, a)][0] for a in factors], axis=0)
    Rall = np.mean([partial281[(drug, a)][1] for a in factors], axis=0)
    Gtwo = 0.5 * (partial281[(drug, 2.0)][0] + partial281[(drug, 0.5)][0])
    Rtwo = 0.5 * (partial281[(drug, 2.0)][1] + partial281[(drug, 0.5)][1])
    errors = (
        float(np.max(np.abs(Gall - stats263["GD"][drug]))),
        float(np.max(np.abs(Rall - stats263["RD"][drug]))),
        float(np.max(np.abs(Gtwo - stats263["GD"][drug]))),
        float(np.max(np.abs(Rtwo - stats263["RD"][drug]))),
    )
    moment_errors285[drug] = errors
    print(drug, "all-five G/R errors; two-factor G/R differences:", errors)

T285 = np.asarray(weighted_fit281(0.3, 0.25, 8.0))
reproduction285 = float(np.max(np.abs(
    T285 - weighted281[(0.3, 0.25, 8.0)]["T"])))
print("Candidate reproduction error", reproduction285)
print("Unit-half-weight versus multi-dose baseline",
      float(np.max(np.abs(weighted_fit281(0.25, 0.25, 1.0)
                          - grid269[(0.25, 0.25)]["T"]))))

assert max(v for e in moment_errors285.values() for v in e[:2]) < 1e-8
assert reproduction285 < 1e-8
assert T285.shape == current284.shape and np.isfinite(T285).all()
assert np.max(np.abs(current284 -
                     grid269[(0.25, 0.25)]["T"])) < 2e-6

backup285 = {name: load(name).clone() for name in edited()}
save_edit("attr_heads.value_bin.2.weight",
          torch.tensor(T285[:-1].T.copy(), dtype=torch.float32))
save_edit("attr_heads.value_bin.2.bias",
          torch.tensor(T285[-1].copy(), dtype=torch.float32))
print("STAGED weighted281 (0.3, 0.25, 8.0)")
print("Offline scores", weighted281[(0.3, 0.25, 8.0)]["score"])

print("=== SCORING HELPERS FOR MULTI-DOSE AUDIT ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and (
        "263" in name or "269" in name
        or "nll228" in obj.__code__.co_names
    ):
        print(name, inspect.signature(obj))
print("stats263 keys", list(stats263))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 285 / rebalanced candidate staged\n"
            f"All-five moment reconstruction errors: {moment_errors285}. "
            f"Candidate reproduction error {reproduction285:.3g}. "
            "Baseline discrepancy reflects replacing five-factor NE constraints with "
            "x2/half constraints, not a failed numerical solve. Saved weighted281 "
            "(.3,.25,8) final value-head weights/bias; other five edited tensors unchanged. "
            "backup285 preserves balanced .25/.25. Offline NLL deltas "
            "+.000727/+.003526/+.002973. GPU factual and rollout validation pending; "
            "additional-dose generalization still needs auditing.\n")
```