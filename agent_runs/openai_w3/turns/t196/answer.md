The reserved head’s value matrix has rank one, leaving other value channels available for a new signal. This means we need not restrict the correction to its original output span. I will check the dose decoder’s factual error and locate the recurrent calibration helper. Small factual signals alongside accurate edited-dose contrasts would justify an unrestricted transport pilot.

```python
print("=== CUSTOM FUNCTION INVENTORY ===")
for k176 in sorted(list(globals())):
    v176 = globals()[k176]
    if inspect.isfunction(v176) and v176.__module__ == "__main__":
        print(k176, inspect.signature(v176))
print("\n=== RESERVED VALUE CHANNELS ===")
print("row norms:", torch.linalg.vector_norm(kv174[336:384],dim=1).numpy())
print("\n=== SIGNAL SUMMARIES ===")
for k176 in ["Y154","payload154","signal169","target170","decoder170",
             "V170","value_carrier171","carrier154","cost154"]:
    v176 = np.asarray(globals()[k176])
    print(k176, v176.shape,
          "mean", np.mean(v176,axis=0).reshape(-1)[:6],
          "sd", np.std(v176,axis=0).reshape(-1)[:6],
          "range", (float(v176.min()),float(v176.max())))
print("decoder factual vs target RMS:",
      np.sqrt(np.mean((V170 @ decoder170 - target170)**2,axis=0)))
print("\n=== CALIBRATION / REPLAY GLOBALS ===")
for k176 in ["orig169","r169","replay_best162","reserved154","sample153",
             "i169","indices155","codes171"]:
    v176=globals()[k176]
    if isinstance(v176,dict):
        print(k176, {k:(tuple(v.shape) if isinstance(v,(torch.Tensor,np.ndarray)) else v)
                     for k,v in v176.items()})
    elif isinstance(v176,tuple):
        print(k176, [(tuple(v.shape) if isinstance(v,(torch.Tensor,np.ndarray)) else type(v).__name__)
                     for v in v176])
    else:
        print(k176, v176 if np.asarray(v176).size<25 else np.asarray(v176).shape)
print("\n=== POSTERIOR CHANGES ===")
for k176,v176 in post154.items():
    old176=load_original(k176)
    print(k176, "delta norm",float((tensor153(v176)-old176).norm()))
with open("/work/notes.md","a") as f:
    f.write("\nPython run 176; forwards 10. Reserved head value matrix is rank one, so its unused value channels permit unrestricted output directions. Checking factual dose-signal error and locating recurrent helper before recalibration. No additional edits.\n")
```