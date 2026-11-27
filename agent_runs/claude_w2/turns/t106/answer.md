The vocab gives each code's median, IQR and fine-bin edges. For the doses this matters: insulin bins saturate above 5 U/h (bin 16). Norepinephrine rates are capped at 1.0 in training, so its rate bins 13 to 16 are never seen there. Doubled norepinephrine (rates up to 2.0) lands in bin 16, a row learned only from other codes. The z-scores stay linear in the dose up to the clip at ±10.

Before writing the clean implementation I need two embedding details I can't see in the lost function sources:
- which categorical tables embed_atoms adds;
- how embed.kq is indexed.

The constants inside those compiled functions (co_consts) will name the tables. Next I print them, plus some END-atom rows of ATM to see which attributes END records carry. If the constants show the table names and the kq index rule, I can write /work/impl.py in the next run and check it against the existing Eb/Ub encodings.

```python
for fnn in ['atom_feats','embed_atoms','bundle_enc','zpost_from','empty_enc','stay_atoms']:
    f=globals()[fnn]
    cs=[c for c in f.__code__.co_consts if not isinstance(c,types.CodeType)]
    print(fnn, cs[:60])
    print('  vars', f.__code__.co_varnames[:40])
print(ATM[ATM.kind==2].head(4).to_string())
print(ATM.groupby(['code','kind','ekind']).size())
```