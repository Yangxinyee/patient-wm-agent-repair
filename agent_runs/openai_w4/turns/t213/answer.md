Smaller amplitudes improved decoding, but held-out dose increments still transmit at only about 80% of the intended size. I won’t compensate by blindly increasing the response weight. Next I’ll separate posterior and decoder errors, check the dose-index alignment, and recover the factual evaluator; that will show where the attenuation comes from.

```python
print("INDEX ALIGNMENT")
print("indices102 == ix180",np.array_equal(indices102,ix180))
print("held active: run87 / run180",
      int((np.asarray(dm87,bool)&np.asarray(te87,bool)).sum()),
      int((np.asarray(active_ne180,bool)&np.asarray(test180,bool)).sum()))
ne_records188 = repair_records.loc[
    repair_records.event.eq("norepinephrine")
]
physical_ne188 = (
    ne_records188.groupby("bundle_index")["rate"].sum()
    .reindex(indices102,fill_value=0).to_numpy()
)
valid188 = np.isfinite(delta185)
print("physical rate vs retained increment RMS",
      np.sqrt(np.mean((physical_ne188[valid188]-delta185[valid188])**2)))
print("physical and retained quantiles",
      np.quantile(physical_ne188[physical_ne188>0],[.05,.5,.95]),
      np.nanquantile(delta185[delta185>0],[.05,.5,.95]))

for label,mask in [
    ("run87",np.asarray(dm87,bool)&np.asarray(te87,bool)),
    ("run180",np.asarray(active_ne180,bool)&np.asarray(test180,bool))
]:
    qbase = gated_ne185(raw87,coef_ne186)
    for f in (.5,2.0):
        qcf = gated_ne185(raw_cf87[("norepinephrine",f)],coef_ne186)
        expected = (f-1)*physical_ne188
        use = mask & (abs(expected)>1e-8)
        change = qcf-qbase
        print("RAW GATED",label,f,"n",int(use.sum()),
              "slope",np.dot(expected[use],change[use])/np.dot(expected[use],expected[use]),
              "RMSE",np.sqrt(np.mean((change[use]-expected[use])**2)))

print("\nFACTUAL EVALUATION HELPERS")
for name,obj in list(globals().items()):
    if callable(obj) and hasattr(obj,"__code__") and any(
        t in name.lower() for t in ("nll","factual","readings","evaluate","head_features")
    ):
        print(name,inspect.signature(obj),"dependencies",obj.__code__.co_names[:24])

print("\nRETAINED REPLAY / READING OBJECTS")
for name,obj in list(globals().items()):
    if not re.search(r"(?:125|126|128|129|131|138|144|161|167)$",name):
        continue
    if isinstance(obj,(np.ndarray,torch.Tensor)):
        print(name,tuple(obj.shape))
    elif isinstance(obj,dict):
        print(name,"dict",list(obj)[:8])
    elif hasattr(obj,"columns"):
        print(name,obj.shape,"columns",list(obj.columns)[:12])

print("\nBASELINE FACTUAL REPLAY")
baseline_replay188 = replay_factual125(capture_decoder=False)
if isinstance(baseline_replay188,dict):
    print("dict",[(k,getattr(v,"shape",type(v).__name__))
                  for k,v in baseline_replay188.items()])
elif isinstance(baseline_replay188,(tuple,list)):
    print(type(baseline_replay188).__name__,
          [(i,getattr(v,"shape",type(v).__name__))
           for i,v in enumerate(baseline_replay188)])
else:
    print(type(baseline_replay188).__name__,getattr(baseline_replay188,"shape",None))

print("\nRECORD-IDENTIFICATION NOTES")
sections188 = re.split(r"(?=\n## )",Path("/work/notes.md").read_text())
chosen188 = [
    s for s in sections188
    if re.match(r"\n## (?:Run|Runs) (?:13[6-9]|14[0-6]|15[4-8])\b",s)
]
print("".join(chosen188)[-3300:])

with open("/work/notes.md","a") as f:
    f.write(
        "\n## Run 188\n"
        "Run-187 amplitude 10/20/40 channels had held-out NE decoding R2 "
        "~.976 and insulin-preservation RMSE ~.028, but dose increment slopes "
        "were only ~.794 (half) and .812 (double). Smaller amplitudes did not "
        "resolve attenuation. Checking physical-rate/index alignment and "
        "posterior versus E-decoder sensitivity before calibration. Recovered "
        "factual evaluation interfaces and replayed the submitted baseline. "
        "No saved weights changed.\n"
    )
```