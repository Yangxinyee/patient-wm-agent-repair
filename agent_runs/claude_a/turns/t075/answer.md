The core-step replica is exact for one variant: errors of 6e-4 against a state scale of 35, and 6e-6 on `out`. The cross-attention computes keys and values from M + age(log(1+age_s)), with age in raw seconds, and the memory includes the current bundle at age 0.

Together with the validated encoders and heads, I now have the whole model on the CPU. I can run teacher-forced passes and dose-edit rollouts offline and check candidate weight edits before spending forward passes.

Next I need to know how to call my bundle encoder. I'll record the result in the notes, then call `encode_bundles` on one stay's atoms to see its input format and outputs. I'll confirm it reproduces `eALL`/`uALL` for those rows, and check whether drug-only bundles have the constant empty measurement vector. If they do, the memory M carries no dose information, and any dose effect must persist through h.

```python
with open('/work/notes.md','a') as fh:
    fh.write("- Run70: CORE STEP REPLICA EXACT (|err| 6e-4 on h scale 35; out 6e-6): hf=h*exp(-dt*exp(flow.a)); prelude tokens [hf+slots, e, u, z_in(zpost)] pre-norm layer -> prelude_norm(slots)=E; "
             "blocks x=hf+E, per layer: self-attn(norm1), cross(norm2(x), kv from M+age_emb(log1p(age_s)) [raw seconds], M includes current e (age0), last128), ffn(norm3); "
             "hn=abar*hf+sp*b_in(E)+(x-(hf+E)); y=c_out(hn); coda pre-norm layer; out=out_norm(mean slots). 4 heads d48, gelu. Functions core_step/mha/ffn/LNp/age_emb in kernel.\n")
print("AT cols", list(AT.columns)); print(AT.head(4).to_string())
print("sub cols", list(sub.columns)); print(sub.head(3).to_string())
print("all_rows[:3]", all_rows[:3], type(all_rows[0]))
print("vis_list", vis_list[:3] if len(vis_list) else vis_list, "| ACT", {k:(type(v).__name__) for k,v in ACT.items()})
co = encode_bundles.__code__; print("encode_bundles consts", co.co_consts, "varnames", co.co_varnames)
co = atom_vectors.__code__; print("atom_vectors consts", [c for c in co.co_consts if not hasattr(c,'co_code')][:60])
```