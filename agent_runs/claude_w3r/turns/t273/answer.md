The tensor layout is simple. The prelude attention has `q` (192×192), `kv` (384×192, presumably K then V) and `out` (192×192), all without biases. `core.b_in` and `core.c_out` are plain 192×192 matrices with no bias. The value head is 192→192(+bias)→17(+bias). The compensation can go into `embed_transition.value_bin` (221 = 13 codes × 17 last bins).

Next, I will:
- confirm the free value dims of head 2 (SVD of its `out` block),
- check whether prelude `norm3` has near-zero gains (dims the prelude FFN ignores, useful for a clean injection direction),
- get the signatures of the kernel helpers I will use for local verification.

If head 2's `out` block has about 3 near-null singular values, I can add a rate read there without changing the original computation.

```python
import inspect
for f in ['step_at','step_parts','run_from','run_stay','core_step_v','wc_row','coda','u_of','prel_heads','prelude_att','n1_of','head_inputs','pred_v','pred_mean_gf','stay_data','ridge_read','rread','att_edit','mixture']:
    try: print(f, inspect.signature(globals()[f]))
    except Exception as ex: print(f,'?',ex)
Wo=load('core.prelude.0.attn.out.weight').double().numpy(); KV=load('core.prelude.0.attn.kv.weight').double().numpy()
Wo2=Wo[:,96:144]; V2=KV[192+96:192+144,:]
U_,S_,Vt_=np.linalg.svd(Wo2); print('Wo2 sv', np.round(S_[:4],3), np.round(S_[-6:],10))
print('V2 sv', np.round(np.linalg.svd(V2,compute_uv=False)[-6:],6))
g3=load('core.prelude.0.norm3.weight').numpy(); gp=load('core.prelude_norm.weight').numpy()
print('norm3 |g| sorted', np.round(np.sort(np.abs(g3))[:8],4), 'prelude_norm |g| min', np.round(np.sort(np.abs(gp))[:5],4))
```