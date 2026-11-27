Run 214 found the drug-bundle counts in the 99 stays: 2,371 of 5,244 bundles hold a drug. It also found several cached DataFrames, including an hourly panel `D2` with 720,000 rows (30,000 stays × 24 hours).

To rebuild z_post's input `[ctx, e, u]` on drug bundles I need:
- the encoder tensor names, to get the right prefix for `setenc_my`;
- a table holding each stay-hour's drug rates and readings. `D2` is the likely candidate.

Next I'll print those names and `D2`'s columns. If `D2` has per-hour rates and readings for the training stays, I can assemble e and u for every S2 drug bundle and check the z posterior rebuild.

```python
print([n for n in NAMES if ('enc' in n or 'set' in n) and n.endswith('weight')][:30])
print('D2', list(D2.columns)); print(D2.head(2).to_string()[:400])
print('F', list(F.columns)[:17])
```