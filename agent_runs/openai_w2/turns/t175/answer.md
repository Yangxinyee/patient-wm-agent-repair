Pre-ICU predictions are unchanged to within 0.0001, confirming that the initial constraint works. Most remaining factual cost is in arterial MAP, and the 99-stay NLL estimate has substantial sampling uncertainty. I’ll prepare an untouched cohort and recover the persistence-fit helpers. Passing on new stays would strengthen the case for this edit.

```python
tuning152 = pd.read_csv("/data/tuning.csv")
all_tuning_ids152 = np.sort(tuning152.stay_id.unique())
unused_ids152 = [
    int(s) for s in all_tuning_ids152
    if int(s) not in set(int(x) for x in cohort150)
]
cohort152 = unused_ids152[:100]
print("Tuning stays:", len(all_tuning_ids152))
print("Untested factual cohort:", cohort152)
print("Untested cohort size:", len(cohort152))

print("\n=== STATISTICAL MODEL HELPERS ===")
keys152 = list(globals())
for name152 in keys152:
    if name152.startswith("_"):
        continue
    obj152 = globals()[name152]
    if callable(obj152) and any(
        s in name152.lower()
        for s in ["kalman", "ssm", "persist", "state", "panel", "em_", "fit"]
    ):
        try:
            print(name152, inspect.signature(obj152))
        except Exception:
            pass

print("\n=== SAVED PERSISTENCE FITS ===")
printed152 = 0
for name152 in keys152:
    obj152 = globals()[name152]
    if isinstance(obj152, dict) and "rho" in obj152 and "Q" in obj152:
        print(name152, {
            k: (np.asarray(obj152[k]).tolist()
                if hasattr(obj152[k], "shape") else obj152[k])
            for k in ["rho", "b", "Q", "R", "drift", "bias"]
            if k in obj152
        })
        printed152 += 1
        if printed152 >= 12:
            break

print("\n=== PANEL OBJECTS ===")
printed_panels152 = 0
for name152 in keys152:
    obj152 = globals()[name152]
    if isinstance(obj152, pd.DataFrame) and any(
        s in name152.lower() for s in ["panel", "train", "hour"]
    ):
        print(name152, obj152.shape, obj152.columns.tolist()[:24])
        printed_panels152 += 1
        if printed_panels152 >= 8:
            break

with open("/work/notes.md", "a") as f152:
    f152.write(
        "\n## Runs 151–152: factual attribution and target audit\n"
        "Initial/pre-ICU mean shifts <=0.0001, NLL changes zero. "
        "Current +0.002976 factual NLL has stay-cluster SE 0.001566 on 99 stays. "
        "Sensor costs: finger +0.002123, lab -0.004625, arterial +0.008312, "
        "cuff +0.001800. Need new-cohort validation. Earlier latent Gaussian "
        "diagnostic gave rho 0.9641 insulin and 0.8445 NE at fixed effects -1.8/+3.6; "
        "targets on the first cohort -14.796/+7.398 and +7.143/-3.572. "
        "Those estimates rely on the statistical state-model assumptions and "
        "must not be treated as known truth.\n"
    )
```
