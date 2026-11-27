Remapping bin 16 to bin 12 changed almost nothing:

| Readout | Original | Remap |
|---|---|---|
| NE x2 | -0.498 | -0.422 |
| Insulin x2 | -1.847 | -1.868 |
| Insulin half, NE half | — | identical |

So the bin embeddings carry little of the dose signal. The out-of-range bin is not why NE x2 is negative. Even with bins unchanged at the cap (12→12), raising only the numeric z past its training range lowers predicted MAP. That is a wrong-signed extrapolation of a confounded response, which input edits can only clamp, not reverse.

I will drop the remap. Next I will log this in the notebook and list the `core.*` tensors (shapes only) outside the blocks. I am looking for any gate or per-dimension parameter on the E→h path (b_in, delta_in, flow) that could scale the action-driven update differently from the reading-driven one.

```python
clear_ops()
with open('/work/notes.md','a') as f:
    f.write("\n## Restart-10 (runs 175-188)\n- Pass 26: combo numeric x5 + head0 x2.5 factual 1.63969 (+0.0051) -> OVER budget. Rejected.\n"
            "- Numeric feats [z_value,has_value,z_dose,has_dose,z_rate,has_rate]; z_dose==z_rate and dose bin==rate bin for both drugs (NE dose=4.8*rate).\n"
            "- NE never in bin 16 in train (cap 1.0); x2 sends 54% NE atoms to bin16. Insulin uses bin16 (0.7%, max 14.5).\n"
            "- Orig NE x2 by band: <=.5 model -.33 (true +.73), .5-1 +.28 (4.57), cap -.70 (6.88). Insulin ratio ~0.14 both bands.\n"
            "- Pass 27: bins row16:=row12 (dose&rate): -1.87/+0.78/-0.42/-0.60 -> bins barely matter. Cleared.\n"
            "- Heads read z from PRIOR (not posterior). LN1 bounds u-token deviation -> input-side gains capped (~-2.6 ins x2).\n")
cn=[n for n in NAMES if n.startswith('core.') and 'blocks' not in n]
print(len(cn)); print([(n,SHAPES[n]) for n in cn])
```