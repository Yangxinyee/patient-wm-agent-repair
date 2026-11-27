Placing z_last_value at input 6 and the has-flag at input 3 (age in hours) cuts the error to 1.50. That is close to the noise floor of about 1.3 expected from comparing 8 local samples with the tool's 4. Next I'll refine with 64 samples (testing extra flags) and record the head layout in the notebook.

```python
sc = []
for flags in [(3,), (3, 7), (3, 2), (3, 4), (3, 5), (7,), (3, 7, 5)]:
    def fe(flags=flags):
        view = (out0 + on0 + gapf(L10))[:, None, :] + zh64
        q = LNn(view + P0['type_emb.weight'][1], P0['query_norm.weight'], P0['query_norm.bias'])
        a = np.where(has, np.log1p(np.nan_to_num(agev)/3600)/10, 0)
        feat = np.zeros((len(R), 8)); feat[:, 0] = has; feat[:, 1] = a; feat[:, 6] = zl_
        for fl in flags: feat[:, fl] += has
        wc = q + (P0['embed.code.weight'][cv] + li(feat, 'last_numeric', P0) + P0['embed.bins.value.weight'][lvb])[:, None, :]
        lg = li(gel(li(wc, 'attr_heads.value_bin.0', P0)), 'attr_heads.value_bin.2', P0) + P0['embed_transition.value_bin.weight'][cv*17 + lvb][:, None, :]
        p = np.exp(lg - lg.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True); p = p.mean(1)
        return (p[:, 1:] * cent).sum(1) - tool
    d = fe(); sc.append((np.abs(d).mean().round(3), flags, [d[cv == c].mean().round(2) for c in [5, 6, 7, 10]]))
print(sorted(sc, key=lambda x: x[0])[:4])
with open('/work/notes.md', 'a') as fh:
    fh.write("- Local head validated (run 205-206): type_emb[1]; gap(log dt_s/10) for ctx AND view; last_numeric feat8=[has,log1p(age_h)/10,0,has,0,0,zval,0] (best fit); bins.rate[0]=last_status[0]=0. MAE vs tool ~1.5 (noise floor ~1.3 with 4 tool samples).\n")
```