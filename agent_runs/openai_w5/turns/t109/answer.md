The finite tests reject these action-injection directions: responses saturate and sometimes have the wrong sign. I won’t use them. I’ll instead examine a targeted value-head correction, which could amplify dose information already present in the state without disrupting recurrence. It must generalize across stays and pass paired factual and rollout tests.

```python
print("=== EXISTING HEAD / REGRESSION OBJECTS ===")
for k96,v96 in list(globals().items()):
    if any(s96 in k96.lower() for s96 in
           ["headtrain", "head5", "cache", "decoder", "ridge", "correction", "head_layout", "hidden"]):
        if isinstance(v96, (np.ndarray,torch.Tensor)):
            print(k96, tuple(v96.shape))
        elif isinstance(v96,dict):
            print(k96, "dict", len(v96), list(v96)[:15])
        elif inspect.isfunction(v96):
            print(k96, inspect.signature(v96), "refs", v96.__code__.co_names)
print("CACHE COLUMNS", list(cacheTrain["df"].columns))
print("CACHE FIRST ROWS", cacheTrain["df"].head(3).to_dict("records"))
print("HEAD LAYOUT", head_layout_spec)
hf96 = head_features_layout(
    cacheTrain, layout=head_layout_spec["layout"],
    ageunit=head_layout_spec["ageunit"], subset=np.arange(200),
    nsamples=4, seed=913)
print("HEAD RETURN TYPE", type(hf96))
if isinstance(hf96,dict):
    for k96,v96 in hf96.items():
        print(k96, getattr(v96,"shape",None), str(v96)[:120] if not hasattr(v96,"shape") else "")
elif isinstance(hf96,(tuple,list)):
    print([(type(v96).__name__,getattr(v96,"shape",None)) for v96 in hf96])
else:
    print(getattr(hf96,"shape",None))
with open("/work/notes.md","a") as f96:
    f96.write("\nRun 95 rejects the large action directions: finite effects saturate and frequently reverse sign. No weights edited. Moving to a GPU-validated, targeted value-head correction.\n")
```
