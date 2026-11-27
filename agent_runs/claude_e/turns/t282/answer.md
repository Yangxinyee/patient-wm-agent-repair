Rejected: the NE pair is far too expensive on random stays.

On tu0, `gated3(-2.7, 2)` costs +.00827 per reading, against +.00459 for `ins_only(-2.7)` on the same stays. The NE pair alone therefore adds about .0037, five times my offline-based estimate. That cost has two sources:

| Row type | Change per row | Share of overall cost |
|---|---|---|
| map_arterial | +.008 | about .0022 |
| glucose_finger and glucose_lab (gate leak) | +.0026 each | about .0010 |
| map_cuff | small | small |

Even at gain 1, the cost would push tu0 past .005. At gains low enough to be safe, the improvement in the NE readouts would be negligible. The NE head readout is mostly noise (R² about .2): any gain large enough to move the rollout readouts costs factual fit on every row where NE is running. NE x2 cannot be reached from the head at all, because it relies on rates the model never saw for NE. So I am dropping NE for good.

Next I will restore `ins_only(-2.7)`, verify it matches the edit used in fwd_26, and update the notebook. Then I will spend a forward pass on tuning batch tu1, the second most expensive batch, to confirm the hash and measure the cost at -2.7 there. A cost below about .0047 keeps -2.7; anything above .005 would make me step back to -2.6.

```python
E = ins_only(-2.7)
for k in ['W0','b0','W2']:
    save_edit(KMAP[k], E[k].float())
print(edited(), 'saved == ins_only(-2.7):', all(np.allclose(load(KMAP[k]).numpy(), E[k].float().numpy()) for k in ['W0','b0','W2']))
o131 = load_original(KMAP['W0']).numpy()[[131,159]]; print('units 131/159 restored:', np.allclose(load(KMAP['W0']).numpy()[[131,159]], o131))
with open('/work/notes.md','a') as f:
    f.write("- fwd_27: dose_edit tu1 with gated3(-2.7,2) (reached rows): ins x2 -12.93 (truth -14.67, .88; ins_only(-2.6) -12.47), half 7.50 (truth 7.34, 1.02; -2.6: 7.19);\n"
            "  NE half -1.10 (orig -.54, truth -2.55), NE x2 +.26 (orig -.11, truth 5.10). NE pair helps both, but...\n"
            "- fwd_28: factual tu0 with gated3(-2.7,2): +.00827 vs ins_only(-2.7) +.00459 -> NE pair costs +.0037 (ma +.008/row, gf/gl +.0026/row leak). 5x offline estimate. NE DROPPED FOR GOOD.\n"
            "- Restored ins_only(-2.7) (units 131/159 original). Next: factual tu1 at -2.7 (confirm hash 2bc751d207, cost; tu1 at -2.5 was .00369). If >.005 -> step back to -2.6.\n")
```