The architecture confirms that the edit changes reading predictions without modifying the recurrent dynamics. The recovery output was truncated before the causal estimates and stay lists appeared. I’ll retrieve those in bounded output so the next rollout tests use the intended batches and record-derived targets.

```python
import re

print("=== CURRENT EDIT INTEGRITY ===")
print("Edited tensors:", edited())
for label in ("amplitude093_checkpoint", "sharp240_checkpoint",
              "validated_gated8_checkpoint"):
    cp = globals().get(label)
    print(label, "available:", isinstance(cp, dict))
    if isinstance(cp, dict):
        print(" maximum current tensor differences:",
              {n: round(float(np.max(np.abs(np.asarray(load(n)) -
                                           np.asarray(cp[n])))), 7)
               for n in edited() if n in cp})

print("=== FACTUAL 41 SCHEMA ===")
print(factual41_original.columns.tolist())
print(factual41_original.head(1).to_string(index=False))
batch_first100 = sorted(factual41_original["stay_id"].unique().tolist())
all_tuning_ids = sorted(pd.read_csv(
    "/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique().tolist())
batch_last100 = all_tuning_ids[-100:]
print("FIRST100:", batch_first100)
print("LAST100:", batch_last100)
print("MIDDLE BATCH RANGES:",
      [(b[0], b[-1], len(b)) for b in
       (all_tuning_ids[100:200], all_tuning_ids[200:300])])

print("=== RECORD-DERIVED EFFECT EVIDENCE ===")
heading = ""
evidence = []
for line in notes_text.splitlines():
    if line.startswith("##"):
        heading = line
    if (line.startswith("-") and
        re.search(r"causal|instrument|target|exclusion|identif|"
                  r"effect|null|placebo|slope|long.horizon|ground.truth",
                  line, re.I)):
        evidence.append(heading + "\n" + line)
print("\n".join(evidence)[:4900])

print("=== EXISTING EFFECT-ESTIMATE VARIABLES ===")
print([k for k in list(globals())
       if not k.startswith("_") and
       any(s in k.lower() for s in
           ("causal", "target", "wald", "iv_result", "rd_result"))][:45])
```