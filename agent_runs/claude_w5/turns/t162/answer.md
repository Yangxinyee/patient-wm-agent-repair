The doc confirms the head structure: `with_code` adds a "last value of the same code" register to the normalised query. That register is the transition table plus the bin and numeric features of the last reading. One gap: `last_numeric` takes 8 inputs but the doc names only 4, and an earlier helper's defaults hint at a layout with the constant in column 0, age in column 3 and z-value in column 6.

To pin down the layout and the type index for glucose and MAP readings, I'll look at:
- the 8 column norms of `last_numeric.weight` (used columns should stand out);
- the `type_emb` row norms;
- the vocab's event/atom kinds and config.

```python
print("last_numeric col norms:", W['last_numeric.weight'].norm(dim=0).numpy().round(3))
print("type_emb row norms:", W['type_emb.weight'].norm(dim=1).numpy().round(2))
print("last_status row norms:", W['last_status.weight'].norm(dim=1).numpy().round(2))
print("event_kinds:", str(VOCAB['event_kinds'])[:300])
print("atom_kinds:", str(VOCAB['atom_kinds'])[:200])
print("codes:", str(VOCAB['codes'])[:400])
print("config:", str(VOCAB['config'])[:500])
```