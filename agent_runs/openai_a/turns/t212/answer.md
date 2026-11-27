Scale 2 also passed the second group’s factual check, with NLL increasing by 0.00308. Much of the cost is in glucose_lab. I’ll inspect the graft’s gate and cached inputs next; restricting the correction to glucose_finger may free factual margin. That idea is supported only if lab perturbations disappear while insulin contrasts remain.

```python
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Independent factual check; forwards 16/50\n"
            "- Second 100 tuning stays: original 1.62044, scale2 1.62352, delta +0.00308.\n"
            "- Scale2 remains a provisional fallback; lab NLL cost motivates a finger-selective gate.\n")
print("CONTRAST CACHE OBJECTS")
for k,v in list(globals().items()):
    if not k.startswith(("sc_", "r3_")):
        continue
    if isinstance(v, dict):
        print(k, "dict", len(v), "keys", list(v.keys())[:9])
    elif isinstance(v, (list,tuple)):
        print(k, type(v).__name__, len(v), repr(v[:3])[:100])
    elif hasattr(v,"shape") and not callable(getattr(v,"shape",None)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v,(float,int,str,bool)):
        print(k, repr(v)[:100])
print("\nGATE / HEAD PARAMETERS")
for k in ["sc_nodes","sc_gate_w","sc_gate_b","sc_rank","sc_bias_comp"]:
    v=globals().get(k)
    if hasattr(v,"shape"):
        a=to_np0(v)
        print(k, a.shape, "min/max/norm",float(a.min()),float(a.max()),float(np.linalg.norm(a)))
    else:
        print(k,repr(v))
print("\nCACHE FUNCTION")
fun=sc_cache_inputs
print(inspect.signature(fun),fun.__code__.co_names,fun.__code__.co_consts)
print("\nEVALUATOR INSTRUCTIONS")
print("\n".join(f"{i.opname} {i.argrepr}" for i in dis.get_instructions(sc_evaluate))[:5200])
print("\nVOCAB CODES", VOCAB["codes"])
print("\nMETADATA COLUMNS", head_meta_train.columns.tolist())
```
