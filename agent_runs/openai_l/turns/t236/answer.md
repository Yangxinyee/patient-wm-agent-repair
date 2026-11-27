The factual logits reconstruct accurately, but my interpretation of the paired-cache tuple failed the alignment check. I’ll resolve its coordinate system before fitting anything. A valid interpretation must match factual inputs and recover unit dose contrasts on the appropriate sensor family.

```python
print("RAW VERSUS X200 RMS:", np.sqrt(np.mean((Xraw204-X200)**2)))
pre213 = Xraw204[..., :192] @ Wbase204.T + bbase204
print("X200[:192] versus preactivation RMS:",
      np.sqrt(np.mean((X200[..., :192]-pre213)**2)))
print("Cached original NLL fit/validation:",
      nll179(Lorig179)[fitmask179].mean(),
      nll179(Lorig179)[valmask179].mean())

for drug213 in ["insulin", "norepinephrine"]:
    tup213 = cases207[(drug213, 2.0)]
    ids213 = tup213[0]
    dr213 = tup213[3]
    own213 = family204[drug213][ids213]
    print("\n", drug213, "own rows", own213.sum(), "all rows", len(ids213))
    print("Direction norm:", np.linalg.norm(directions207[(drug213, .1)]))
    for pos213 in [1, 2]:
        x213 = tup213[pos213]
        print(" tuple position", pos213,
              "last coordinate range", (x213[..., -1].min(), x213[..., -1].max()),
              "RMS", np.sqrt(np.mean(x213**2)),
              "gate own mean", (x213[:, own213] @ gatecanon200).mean())
        for label213, ref213 in [
            ("Xraw204", Xraw204[:, ids213]),
            ("X200", X200[:, ids213]),
            ("preactivation+1", np.concatenate(
                [pre213[:, ids213], np.ones((8, len(ids213), 1))], axis=-1)),
        ]:
            print("   RMS versus", label213, np.sqrt(np.mean((x213-ref213)**2)))
        y213 = x213[:, own213] @ directions207[(drug213, .1)]
        print("   direction projection quantiles:", np.quantile(y213, [0, .1, .5, .9, 1]))
        print("   projection/rate slope:",
              np.sum(y213 * dr213[own213][None, :]) /
              (8*np.sum(dr213[own213]**2)))
    print("own rate quantiles:", np.quantile(dr213[own213], [0, .1, .5, .9, 1]))

print("\nCanonical/paired tuple equality:")
for pos213 in range(4):
    a213 = canon_cases200[("insulin", 2.)][pos213]
    b213 = cases207[("insulin", 2.)][pos213]
    print(pos213, "shape", a213.shape,
          "RMS difference", np.sqrt(np.mean((a213-b213)**2)))
print("\nUseful unsuffixed cache variables:")
for name213 in ["Xf", "Xi", "DX", "dx", "xf", "xi", "factual", "selected",
                "tr", "va", "target", "beta", "gate", "rows", "ids"]:
    if name213 in globals():
        print(name213, describe210(globals()[name213]))
```
