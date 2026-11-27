The reversed-transmission variants have reasonable local losses, but I used an intermediate model’s loss array as the reference. I’ll correct that comparison and screen slightly stronger insulin variants before saving one. It must leave factual-loss margin and then pass the GPU rollout test.

```python
original290 = {n: load_original(n).clone() for n in NAMES}
original_nll290 = nll_scalar289(factual_eval194(original290))
print("TRUE CPU ORIGINAL NLL", original_nll290)
print("CORRECTED RUN-289 DELTAS",
      {p: r["nll"] - original_nll290 for p, r in screen289.items()})

params290 = [
    (3.5, 0.0, 26.0, -0.4),
    (4.0, 0.0, 27.0, -0.4),
    (4.0, 0.0, 26.0, -0.4),
    (3.0, 0.0, 26.0, -0.4),
]
candidates290 = {p: candidate286(*p) for p in params290}
screen290 = {}
try:
    for p, w in candidates290.items():
        nll = nll_scalar289(factual_eval194(w))
        screen290[p] = {"nll": nll, "delta": nll - original_nll290}
    print("CPU SCREEN", screen290)
finally:
    factual_eval194(restored289)

preference290 = [params290[0], params290[1], params290[3]]
acceptable290 = [p for p in preference290 if screen290[p]["delta"] <= 0.0042]
if acceptable290:
    chosen290 = acceptable290[0]
    weights290 = candidates290[chosen290]
    backup290 = {n: load(n).clone() for n in edited()}
    for n in NAMES:
        if torch.equal(weights290[n], original290[n]):
            if n in edited():
                drop_edit(n)
        else:
            save_edit(n, weights290[n])
    factual_eval194(weights290)
    print("PROVISIONALLY SAVED", chosen290, "EDITS", edited())
else:
    chosen290 = None
    print("NO NEW CANDIDATE SAVED")

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 290\nCorrected Run-289 reference: baseline_losses194 "
            "was an intermediate-model loss array, not the original. "
            "True same-fixture original CPU NLL = "
            + str(original_nll290) + ". Corrected prior deltas = "
            + str({p: r["nll"] - original_nll290
                   for p, r in screen289.items()}) +
            ". New screen = " + str(screen290) +
            ". Provisionally saved " + str(chosen290) +
            "; GPU factual and rollout validation pending.\n")
```