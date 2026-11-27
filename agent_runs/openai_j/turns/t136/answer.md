That fit did not materially strengthen either drug response, and several settings overfit factual readings. I won’t install it. I’ll remove the factual-label fitting term and instead penalize any factual logit change; useful held-out dose sensitivity with a small loss change would support this approach.

```python
r123_models = {}
r123_results = []
for fw in [1e-5, 1e-4, 1e-3, 1e-2]:
    for ridge in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
        penalty = np.eye(r122_F.shape[1])*ridge
        penalty[-1, -1] *= 0.01
        coef = solve(r122_Gc + fw*r122_Gf + penalty,
                     r122_bc, assume_a="pos")
        tau = r122_F@coef
        Pnew = r122_tilt(r122_P, tau)
        dnll = (np.log(np.maximum(r122_P[np.arange(len(Pnew)), r122_bins], 1e-30))
                - np.log(np.maximum(Pnew[np.arange(len(Pnew)), r122_bins], 1e-30)))
        rec = {"fw": fw, "ridge": ridge,
               "cv_dNLL": dnll[r122_test].mean(),
               "cv_tilt_rms": np.sqrt(np.mean(tau[r122_test]**2)),
               "linear_cv_error": np.sqrt(np.average(
                   (r122_X[~r122_cf_fit]@coef-r122_y[~r122_cf_fit])**2,
                   weights=r122_cw[~r122_cf_fit]))}
        base_new_mu = (Pnew*r122_centres).sum(1)
        for (drug, factor), cf in r117_cf.items():
            Pcn = r122_tilt(cf["probs"], r122_cf_F[(drug, factor)]@coef)
            shift = (Pcn*r122_centres).sum(1)-base_new_mu
            sensor = "glucose_finger" if drug=="insulin" else "map_arterial"
            m = cf["active"] & r122_test & (head_meta.event.to_numpy()==sensor)
            dr = cf["delta_rate"][m]
            rec[f"{drug[0]}{factor}_slope"] = np.dot(dr, shift[m])/np.dot(dr, dr)
        r123_results.append(rec)
        r123_models[(fw, ridge)] = coef
r123_results = pd.DataFrame(r123_results)
print(r123_results.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
print("\n=== EARLIER INPUT/STATE PROBES ===")
r123_notes = Path("/work/notes.md").read_text()
r123_start = r123_notes.find("## Run 37:")
print(r123_notes[r123_start:r123_start+2300])
print("\nActive edits:", edited())
```
