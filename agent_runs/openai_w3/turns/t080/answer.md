The records support a negative insulin effect and a positive norepinephrine effect, with effects persisting into later hours. The original rollouts are much weaker and give the wrong sign for doubled norepinephrine. I’ll inspect the retained state-replay helpers next; an effective repair must change dose sensitivity while leaving ordinary-record predictions nearly unchanged.

```python
import inspect, dis
print("STATE/ENCODING ARRAYS")
for k,v in list(globals().items()):
    if k.startswith("_") or not isinstance(v,(np.ndarray,torch.Tensor,pd.DataFrame)):
        continue
    s = tuple(v.shape)
    if (len(s)>1 and (192 in s or 576 in s or 992 in s or s[0] in (4313,1697))) or k.startswith(("r66","probe","C","S","U","E","Xpost")):
        print(k, type(v).__name__, s)
print("\nSMALL OBJECT NAMES")
print([k for k,v in list(globals().items()) if not k.startswith("_")
       and isinstance(v,(dict,list,tuple)) and k not in ("VOCAB","SHAPES","NAMES")
       and ("state" in k.lower() or "probe" in k.lower() or "repair" in k.lower()
            or "fit" in k.lower() or "action" in k.lower() or "66" in k)])
print("\nHELPERS")
for fname in ["r66_prelude","clean_single_encode","encode_one_set","context_with_flow","make_atom","repair_action_vectors"]:
    f = globals().get(fname)
    if not callable(f):
        continue
    print(fname, inspect.signature(f), "names=", f.__code__.co_names,
          "constants=", repr(f.__code__.co_consts)[:350])
    try:
        print(inspect.getsource(f)[:2300])
    except (OSError,TypeError):
        pass
print("\nRELEVANT TENSORS")
print([(n, SHAPES[n]) for n in NAMES if n.startswith(
    ("core.prelude","core.b_in","core.c_out","core.out_norm","attr_heads.value","z_in","last_numeric"))])
print("\nEFFECT SUMMARY")
for k,v in impulse_results.items():
    print(k, "kernel:", np.round(v["theta"],4).tolist())
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 70; forwards 7. Recovered evidence: insulin immediate effect about -1.94/unit with persistent kernel; NE about +3.2/rate with shorter persistence. Original tuning rollout reference targets: insulin x2 -12.96, half +6.48; NE x2 +4.28, half -2.14. Inspecting retained state/prelude helpers; no edits.\n")
```