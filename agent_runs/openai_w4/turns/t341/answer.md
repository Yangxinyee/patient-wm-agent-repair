This candidate passes factual fit on the 98 stays, with an NLL increase of 0.00364 nats, but the norepinephrine reversal did not work. I’ll compare the earlier norepinephrine construction with modestly increased gain. It would help only if doubled-norepinephrine response becomes positive while insulin response and factual fit remain acceptable.

```python
backup291 = {n: load(n).clone() for n in edited()}
full291 = {n: load(n).clone() for n in NAMES}

params291 = [
    (3.0, 0.0, 26.0, 1.0),
    (3.0, 0.0, 26.0, 1.1),
    (3.0, 0.0, 26.0, 1.4),
    (3.0, 0.0, 27.0, 1.4),
]
candidates291 = {
    p: asymmetric_ins279(p[0], p[1], p[2], ne_gain=p[3])
    for p in params291
}
screen291 = {}
try:
    for p, w in candidates291.items():
        nll = nll_scalar289(factual_eval194(w))
        screen291[p] = {"nll": nll, "delta": nll - original_nll290}
    print("EARLIER-NE CPU SCREEN", screen291)
finally:
    factual_eval194(full291)

print("ACTION FIXTURE VARIABLES",
      [n for n in sorted(globals()) if n.endswith("286")])
for drug, values in actions286.items():
    print("ACTION FIXTURE", drug, values.shape)
print("ACTION HELPERS")
for n in sorted(list(globals())):
    v = globals()[n]
    if callable(v) and any(s in n.lower() for s in ("fixture", "atom", "action")):
        try:
            print(n, inspect.signature(v))
        except Exception:
            pass

preference291 = [params291[1], params291[0], params291[3]]
acceptable291 = [p for p in preference291 if screen291[p]["delta"] <= 0.0045]
chosen291 = acceptable291[0] if acceptable291 else None
if chosen291 is not None:
    weights291 = candidates291[chosen291]
    for n in NAMES:
        if torch.equal(weights291[n], original290[n]):
            if n in edited():
                drop_edit(n)
        else:
            save_edit(n, weights291[n])
    factual_eval194(weights291)
print("PROVISIONALLY SAVED", chosen291)

with open("/work/notes.md", "a") as f:
    f.write("\n## GPU passes 35–36; Run 291\n"
            "Run-290 candidate rollout: insulin -15.571/+9.247, "
            "NE -1.128/-4.164. GPU factual delta +.00364 on 98 stays. "
            "Backed up as backup291. Screened earlier NE construction: "
            + str(screen291) + ". Provisionally saved "
            + str(chosen291) + "; rollout unvalidated. "
            "Record-based effect estimates unchanged.\n")
```