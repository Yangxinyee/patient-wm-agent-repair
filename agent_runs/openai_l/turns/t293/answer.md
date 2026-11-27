The grid confirms interference: increasing insulin’s fitting weight reduces norepinephrine’s fitted response. I’ll test two closed-form restrictions—linear, opposite-signed readout weights for each drug’s feature pair, and limits on dose-response changes through the older features. They would help if both responses strengthen at the same factual-loss cost.

```python
st260 = st242
u260 = st260["arch"]["units"]
s260 = st260["scale"]
t0260 = st260["T0"]
gI260 = st260["GD"]["insulin"]
gN260 = st260["GD"]["norepinephrine"]

a260 = st260["GF"] + 0.5*gI260 + 0.4*gN260 + 1e-4*np.eye(193)
r260 = st260["RF"] + 0.5*st260["RD"]["insulin"] + 0.4*st260["RD"]["norepinephrine"]
check260 = t0260 + np.linalg.solve(a260, r260)/s260[:, None]
print("Current-fit reconstruction error:",
      float(np.max(np.abs(check260-grid242[(0.5, 0.4)]["T"]))))
print("Base coefficients at replaced units:",
      np.linalg.norm(t0260[u260], axis=1))

keep260 = [j for j in range(193) if j not in set(u260)]
p260 = np.zeros((193, len(keep260)+2))
p260[keep260, np.arange(len(keep260))] = 1.0
for j260 in range(2):
    pair260 = u260[2*j260:2*j260+2]
    vals260 = s260[pair260] * np.array([1.0, -1.0])
    p260[pair260, len(keep260)+j260] = vals260/np.linalg.norm(vals260)

m260 = np.ones(193)
m260[u260] = 0.0
protect260 = (gI260+gN260)*m260[:, None]*m260[None, :]
trial260 = {}
for mode260, proj260 in (("free", np.eye(193)), ("linear_pairs", p260)):
    for ei260, en260 in ((0.5, 0.4), (0.5, 0.7), (0.7, 0.7), (1.0, 1.0)):
        for beta260 in (0.0, 0.3, 3.0):
            aa260 = (st260["GF"] + ei260*gI260 + en260*gN260
                     + beta260*protect260 + 1e-4*np.eye(193))
            rr260 = (st260["RF"] + ei260*st260["RD"]["insulin"]
                     + en260*st260["RD"]["norepinephrine"])
            dt260 = proj260 @ np.linalg.solve(proj260.T@aa260@proj260, proj260.T@rr260)
            tt260 = t0260 + dt260/s260[:, None]
            sc260 = score234(st260["arch"], tt260, st260["Ftune"])
            key260 = (mode260, ei260, en260, beta260)
            trial260[key260] = {"T": tt260, "score": sc260}
            print(key260, "NLL", np.round(sc260[:3], 6),
                  "slopes", np.round(sc260[3], 3))

with open("/work/notes.md", "a") as f260:
    f260.write("\n## Run 260 / dose-constraint coupling\n"
               "Recovered the analytic fit. Existing grid confirms stronger insulin weighting "
               "reduces the NE response. Tested antisymmetric GELU-pair readouts and penalties "
               "on dose-contrast changes through older features, using closed-form regressions. "
               "No weights saved; current .5/.4 candidate remains staged.\n")
```
