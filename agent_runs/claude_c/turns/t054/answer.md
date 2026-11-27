The states pass locates the bottleneck. With the z columns ×5, the state's sensitivity to the insulin dose innovation grew only about 1.3× (out 1.28, h 1.29, z 1.40). That matches the 1.36× growth of the glucose response, so the path from state to prediction is roughly linear. The amplification is lost before the state.

Either the z-score columns carry only a small part of the dose information (the shared rate and dose bin tables carry the rest), or the action encoder saturates. The earlier asymmetric increments hint at some saturation as well (x2 −0.66 vs x1/2 +0.88).

For both drugs, the rate and dose of an atom always fall in the same bin index, and z_dose equals z_rate. So the whole dose code is (w2+w4)·z plus (bins.dose+bins.rate)[bin].

Next test: amplify the dose-varying part of both bin tables ×5 together with the z columns ×5. I use a rank-1 row-space edit along a centred profile linear in insulin dose, so the common bin component is untouched. Then I check with a dose-edit pass.

- If the insulin response grows about 5×, the encoding is the lever, and I only need calibration plus a factual check.
- If it grows much less, the encoder saturates, and I need a lever after it.

```python
clear_ops()
# insulin dose represented by each fine bin (rates are multiples of 0.5); unused-by-insulin bins interpolated
d_bin = np.array([np.nan, 0.5, 0.667, 0.833, 1.0, 1.125, 1.25, 1.375, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.75, 6.0])
v_prof = np.zeros(17); v_prof[1:] = d_bin[1:] - d_bin[1:].mean()
print("profile", v_prof.round(2), "zero-point dose", d_bin[1:].mean().round(2))
e2 = np.zeros(6); e2[2] = 1; e4 = np.zeros(6); e4[4] = 1
rank1('embed.numeric.weight', e2, 4.0, side='in')
rank1('embed.numeric.weight', e4, 4.0, side='in')
rank1('embed.bins.dose.weight', v_prof, 4.0, side='out')
rank1('embed.bins.rate.weight', v_prof, 4.0, side='out')
print(ops())
```