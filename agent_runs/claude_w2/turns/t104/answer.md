All tensor names are now known. z_post maps 576→192→64 (mean and log-sd), z_prior maps 192→192→64, and z_in and z_head map 32→192. The vocab note confirms the predicted mean is Σ_{b=1..16} p(b)·centre[b−1].

My earlier embedding details (unit-table rows, rate-unit rows, bin side, END-atom routing) are stored only as default arguments of functions whose source is lost. So I will read them from the function signatures and defaults, plus the atom table ATM and the `variants` dict, rather than rediscover them. Then I can write a clean implementation file. If the defaults show the mappings, such as glucose unit row 3, MAP row 2 and END atoms routed to e, I can encode bundles exactly and go straight on to the core step.

```python
for fnn in ['atom_feats','embed_atoms','bundle_enc','zpost_from','empty_enc','z_check2','z_check3','z_check4','zerr_custom','stay_atoms','atom_vec2','ctx_v']:
    f=globals()[fnn]
    print(fnn, str(inspect.signature(f))[:400])
    print('   names:', f.__code__.co_names[:40])
print(ATM.head(12).to_string())
print(list(variants.keys()))
print(U_codes, U_alpha, RU1, RU2, U_c2, U_c3)
```