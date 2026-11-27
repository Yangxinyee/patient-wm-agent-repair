The original weights are restored. The architecture confirms that doses can affect the recurrent state directly through the action token, without relying on the weak latent pathway. I’ll recover the record-derived effects and available feature arrays next; these will let us test a direct correction rather than guess its direction or strength.

```python
print("EARLY EVIDENCE\n" + open("/work/notes.md").read()[:5700])
print("\nEFFECT FITS")
for key in ["dml_results", "robust_results", "impulse_results", "summary"]:
    obj = globals().get(key)
    if isinstance(obj, dict):
        for k, v in obj.items():
            text = repr(v)
            print(key, k, text[:650])
print("\nRETAINED FEATURE/STATE OBJECTS")
_items69 = list(globals().items())
print(" | ".join(f"{k}:{tuple(v.shape)}" for k,v in _items69
    if not k.startswith("_") and isinstance(v,(np.ndarray,torch.Tensor,pd.DataFrame))
    and k not in {"train","tuning","df","sub","y0","y1","d","ok","x","tab","show","diff",
                  "sid_rows","hour_rows","arr","last","age","total","count","seen","Xhist",
                  "fold_by_stay","fold_rows","dose","next_y","rows","xx","dd","yy","pred_d",
                  "pred_y","fit","test","rd","ry","positive","sel","predictions","target",
                  "r2","rr","quadratic","bread","errors","qse","edges","pdose","mask",
                  "both_now","pred","base_rows"}))
print("\nRELEVANT TENSORS")
print([(n, SHAPES[n]) for n in NAMES if
       n.startswith(("core.prelude","core.b_in","core.c_out","core.out_norm","attr_heads.value","z_in","last_numeric"))])
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 69; forwards 7. Original model restored and architecture recovered. The direct action-token/recurrent pathway is the candidate target; recovering the record-derived causal effects and retained design matrices before calibration.\n")
```