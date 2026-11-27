The cached FFN output weights are original weights, not the installed compensation; using them would misdiagnose the repair. I’ll now fit a small treatment-contrast decoder on the fixed FFN features. It must predict physical dose changes while remaining small on factual records; poor held-out performance would rule out this approach.

```python
print("PAIR CONSTRUCTION")
for name241 in ["r214_parts", "action_vectors87", "hidden192", "posterior153"]:
    f241 = globals()[name241]
    print(name241, inspect.signature(f241), "globals", f241.__code__.co_names)
print("\nr214_parts")
for ins241 in dis.get_instructions(r214_parts):
    if ins241.opname not in ["RESUME","CACHE","NOP","PRECALL"]:
        print(ins241.opname, ins241.argrepr, end="; ")
print()

print("\nDOSE ARRAY")
a241 = as_numpy(rate154) if isinstance(rate154,torch.Tensor) else np.asarray(rate154)
print("rate154", a241.shape, a241[:8],
      "range", np.nanmin(a241,axis=0), np.nanmax(a241,axis=0))
print("\nFEATURE CONSISTENCY")
print("H vs actual hidden RMS",
      float(np.sqrt(np.mean((np.asarray(H220)-arr183(h240).reshape(-1,768))**2))))
print("E vs original RMS",
      float(np.sqrt(np.mean((arr183(parts240[0])-np.asarray(orig_Eflat220).reshape(-1,4,192))**2))))
print("masks", np.asarray(tr220).dtype, int(np.asarray(tr220).sum()),
      np.asarray(va220).dtype, int(np.asarray(va220).sum()))
for k241, v241 in contrast_features220.items():
    a241 = np.asarray(v241)
    print(k241, "last", np.unique(a241[:,-1])[:4],
          "rms", float(np.sqrt(np.mean(a241[:,:-1]**2))))
print("\nDIRECT PAIR VARIABLE METADATA")
for k241 in r214_parts.__code__.co_names:
    if k241 not in globals():
        continue
    v241 = globals()[k241]
    if isinstance(v241,dict):
        print(k241, [(str(k),tuple(v.shape) if hasattr(v,"shape") else type(v).__name__)
                     for k,v in list(v241.items())[:10]])
    elif hasattr(v241,"shape"):
        print(k241, tuple(v241.shape))
with open("/work/notes.md","a") as f:
    f.write("\nPython run 241; forwards 16. Found cached w2_220/b2_220 are original weights; installed factual raw RMS differs. Verifying fixed attention-feature contrasts for a small treatment-innovation decoder and rank-two FFN adjustment. No weights changed.\n")
```