The pilot barely changed the dose responses and still exceeded the factual-loss budget. I’m rejecting it and restoring the original weights. Next I’ll examine a direct dose-innovation channel rather than relying on the fitted FFN features. It must track dose changes across factors without adding substantial signal on ordinary factual bundles.

```python
for name142 in pilot141_names:
    drop_edit(name142)
print("RESTORED:", edited())
print("\nLATENT AND ACTION TENSORS")
for name142 in NAMES:
    if name142.startswith(('z_post.', 'z_prior.', 'z_in.', 'z_head.', 'encode_action.')):
        print(name142, SHAPES[name142])
print("\nRETAINED ENCODING/CHANNEL ARRAYS")
for k in sorted(list(globals())):
    if (k.endswith(('80','81','82','88','104','105')) or
        any(s in k.lower() for s in ['gate104','decoder','payload','innovation'])):
        v = globals()[k]
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict keys", list(v)[:12])
        elif inspect.isfunction(v):
            print(k, str(inspect.signature(v)))
print("\nFORWARD FILES")
for d142 in ['fwd_3','fwd_4','fwd_5','fwd_6','fwd_7']:
    print(d142, os.listdir('/work/'+d142))
with open('/work/notes.md','a') as f:
    f.write("\nPython run 142; forwards 9. Rank-two FFN pilot rejected and dropped. "
            "At gain [1,1], factual NLL rose 1.63982->1.64743 (+.00761), exceeding budget. "
            "Actual rollout shifts were -0.570,+0.194,-0.721,-0.303, versus originals "
            "-0.422,+0.213,-0.742,-0.238. Thus local finite-difference calibration "
            "did not translate into meaningful rollout repair. All weights restored. "
            "Investigating an explicit latent/action dose-innovation transport channel.\n")
```
