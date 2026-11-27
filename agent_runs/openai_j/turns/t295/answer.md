The best saved nonlinear candidate has essentially the same held-out doubling sensitivity as the linear edit, so it does not justify a rollout test. The diagnostic exception occurred during printing and changed no weights. I’ll review the causal-identification evidence and earlier action-path experiments to see whether a genuinely different correction remains available.

```python
print("=== RECORD-DERIVED TARGETS ===")
kt257 = pd.read_csv("/work/kernel_targets.csv")
print(kt257.to_string(index=False))

print("\n=== DYNAMIC IDENTIFICATION RESULTS ===")
di257 = pd.read_csv("/work/dynamic_iv.csv")
print("Columns:", list(di257.columns))
print(di257.head(10).to_string(index=False))

print("\n=== RELEVANT NOTEBOOK SECTIONS ===")
sections257 = re.split(r"(?m)^## ", Path("/work/notes.md").read_text())
wanted257 = []
for i, section in enumerate(sections257):
    title = section.splitlines()[0] if section else ""
    if any(k in title.lower() for k in
           ["posterior", "action", "recurrent", "kernel",
            "instrument", "identification", "dynamic"]):
        wanted257.append((i, section))
print("Relevant headings:", [(i, s.splitlines()[0]) for i, s in wanted257])
chosen257 = wanted257[:5] + wanted257[-3:]
seen257 = set()
for i, section in chosen257:
    if i not in seen257:
        print("\nSECTION", i)
        print(section[:650])
        seen257.add(i)

print("\n=== CURRENT CHECKPOINT CONFIRMATION ===")
safe257 = np.load("/work/safer_candidate247.npz")
for name in edited():
    x = load(name)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    print(name, "matches safer checkpoint:",
          bool(np.array_equal(np.asarray(x), safe257[name])))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 256–257: nonlinear rejection and causal audit\n"
            "- Forward 32 / fwd_33 current edit: insulin -10.251/+5.482; NE +0.899/-1.573.\n"
            "- Current pooled tuning factual dNLL +0.00357436 across 400 stays.\n"
            "- Saved nonlinear scalar candidates provide negligible held-out doubling improvement; not installed.\n"
            "- Run 256 failed only in diagnostic printing; no weights changed.\n"
            "- Reviewing identification and prior action-path attempts before choosing a distinct experiment.\n")
```