The output is still a repeat of run 96, so I cannot tell whether the later test executed. I’ll check the kernel for its results before running anything else. Existing test objects would let us resume; their absence would confirm that the test still needs to run.

```python
print("EXECUTION CHECK AFTER CONFIRMED RUN 96")
for name in [
    "candidate97", "states_local97", "phi_local97", "prob_local97",
    "pair_local97", "rows97", "feature_decoders97"
]:
    obj = globals().get(name)
    print(name, "present" if name in globals() else "ABSENT",
          getattr(obj, "shape", None))
if "rows97" in globals() and isinstance(rows97, list) and rows97:
    print(pd.DataFrame(rows97).to_string(index=False))
if "nll_local97" in globals() and "nll_orig97" in globals():
    print("Local NLL difference:", float(np.mean(nll_local97 - nll_orig97)))
print("Current saved edits:", edited())
print("Notebook tail:")
print(Path("/work/notes.md").read_text()[-1200:])
```
