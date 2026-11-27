The quieter insulin fits have a drawback: they weaken half-dose sensitivity and increase sensitivity to NE edits. Lower residuals alone are therefore not enough. I’ll compare constrained fits that preserve the existing dose responses with fits that make the half-dose response proportional; acceptable residuals under those constraints would justify a rollout test.

```python
probe_means206 = {
    key: D147[probe_labels202 == j].mean(axis=0)
    for j,key in enumerate(probe_keys171)
}
print("CURRENT MEAN PROBE RESPONSES")
print(pd.DataFrame(
    [probe_means206[key] @ Bcurrent202 for key in probe_keys171],
    index=[str(key) for key in probe_keys171],
    columns=["insulin_decoder","NE_decoder"]
).to_string())
print("RECORD RATE-CHANGE TARGET MEANS",
      {key:Y147[probe_labels202==j].mean(axis=0).tolist()
       for j,key in enumerate(probe_keys171)})

Ains206 = np.vstack([
    C147,
    probe_means206[("insulin",2.0)],
    probe_means206[("insulin",0.5)],
    probe_means206[("norepinephrine",2.0)],
    probe_means206[("norepinephrine",0.5)]
])

def constrained_small_fit206(Hf,Hp,g,A,t,pw,ridge):
    H = Hf + pw*Hp + ridge*penalty171
    sol = solve(H,np.column_stack([pw*g,A.T]),assume_a="pos")
    unconstrained, HA = sol[:,0],sol[:,1:]
    b = unconstrained + HA @ np.linalg.pinv(
        A@HA,rcond=1e-10
    ) @ (t-A@unconstrained)
    return b

targets206 = {}
for half_mode206 in ["preserve_half","linear_half"]:
    for cross_mode206 in ["preserve_cross","zero_cross"]:
        half206 = (
            float(current_probe205[ins_half205,0].mean())
            if half_mode206 == "preserve_half" else -0.5*target_ins205
        )
        cross_double206 = (
            float(current_probe205[ne_double202,0].mean())
            if cross_mode206 == "preserve_cross" else 0.0
        )
        cross_half206 = (
            float(current_probe205[ne_half202,0].mean())
            if cross_mode206 == "preserve_cross" else 0.0
        )
        targets206[(half_mode206,cross_mode206)] = np.r_[
            np.zeros(4),target_ins205,half206,cross_double206,cross_half206
        ]

keys206 = [(0.003,1e-6),(0.03,1e-6),(0.1,1e-5),
           (0.3,1e-4),(1.0,1e-4),(3.0,1e-3)]
constrained_fits206,rows206 = {},[]
for mode206,t206 in targets206.items():
    for key206 in keys206:
        b206 = constrained_small_fit206(
            Hpool202,Hp171,Gp171[:,0],Ains206,t206,*key206
        )
        constrained_fits206[(mode206,key206)] = b206.copy()
        rows206.append({
            "mode":"/".join(mode206),"pw":key206[0],"ridge":key206[1],
            "old_RMS":rms202(X147@b206),
            "new_RMS":rms202(X202@b206),
            "constraint_error":float(np.max(np.abs(Ains206@b206-t206)))
        })
table206 = pd.DataFrame(rows206)
print("\nCONSTRAINED INSULIN FITS")
print(table206.to_string(index=False,float_format=lambda x:f"{x:.6g}"))
print("CURRENT INSULIN RMS OLD/NEW",
      rms202(X147@Bcurrent202[:,0]),rms202(X202@Bcurrent202[:,0]))
print("NO WEIGHT CHANGE",
      np.array_equal(np.asarray(load("core.b_in.weight")),candidate_B204))

with open("/work/notes.md","a") as f:
    f.write("\n## Run206: insulin sensitivity constraints\n"
            "Unconstrained zero-factual fits reduce insulin residuals but weaken "
            "half-dose sensitivity and increase cross-drug sensitivity. Comparing "
            "small constrained regressions that preserve or proportionally adjust "
            "half-dose sensitivity and preserve or remove cross-decoder response. "
            "No claim that proxy residual improvements establish causal accuracy; "
            "rollouts and record evidence remain required. Weights unchanged.\n")
```
