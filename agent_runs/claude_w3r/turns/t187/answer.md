Run 174 shows many leftover arrays. `UX` (9918×192) and its partners `YXi`/`YXn` are the 192-dimensional u vectors (insulin/norepinephrine action encodings) of the 9918 bundles in the 100 selS train stays, so the run-170 check was a linear probe of the rates on u. The probe reads both rates almost exactly (slope ≈ 1) in all four dose-edited conditions. So the clean dose signal is available at the u token. What's missing is an amplified path from there to the glucose and MAP heads.

The black-box functions are my local core step, which was already validated, but their source is lost. Before I build on them, I need to know which globals and weight dicts they read: are the weights in `W`, so that I could simulate an edit by patching `W`? I'll inspect their bytecode names. If `W` appears in core_step and step_parts, I can simulate edits locally. If not, I'll need to rewrite the step with saved source.

```python
import dis
for fn in [core_step, core_step_v, core_step_m, core_step_p, step_parts, step_at, run_stay, stay_data, head_feats, head_inputs, pred_mean_gf, pred, pred_c, pred_v, u_of, s_of, eval_stay_rel, score, score2, enc_eu, atom_vec, mha, ffn, LNt, mlp_t, age_emb]:
    c=fn.__code__
    print(fn.__name__, c.co_varnames[:c.co_argcount], 'defaults', fn.__defaults__, '| names:', ' '.join(c.co_names)[:400])
print(type(W), list(W.keys())[:5] if isinstance(W,dict) else '', len(W) if hasattr(W,'__len__') else '')
print(type(Wpre), list(Wpre.keys())[:5] if isinstance(Wpre,dict) else '', type(Wt), list(Wt.keys())[:5] if isinstance(Wt,dict) else '')
```