Two clean channels exist.

**Rate write.** The insulin start rate is linearly decodable from the normalised dose token LN1(u) with test R² 0.995 (residual sd 0.07 on a deviation signal of sd 0.59). The same map gives ≈0 for NE-only bundles (sd 0.01) and exactly 0 for end-only bundles. Prelude head 3 copies u with attention weight 1.0 at insulin bundles. So head 3's value, read through one row of the prelude output projection, can write the rate into a dedicated "staging" dimension. LayerNorm adds only a multiplicative ±11%; a reference dimension cancels the common-mode term.

**Baseline write.** At glucose-reading bundles, head 2 attends to the measurement token e with weight 0.93 (sd 0.25), but head 3 only 0.63 (sd 0.47). So the −rbar(tier) baseline route is noisier. I will fit it on the actual head outputs.

**NE needs care.** Reasoning it through: any NE injection that responds to dose levels on factual records double-counts what the model already uses as a severity proxy, and would cost far more than 0.005 nats. Only the within-step titration noise is in the model's blind spot. I will analyse NE after insulin works.

**Next step:**
1. Save these findings to the notebook.
2. Reconstruct the prelude tokens and head values offline, validating against the captured attention weights.
3. Regress the insulin write target on head-3 (and head-2) outputs.

**What would confirm or refute it:**
- Reconstructed attention matches (error ≈ 0), and the rate write via head 3 has residual sd ≲ 0.1: then the write side of the design is settled.
- A poor baseline fit means the −rbar part needs another route.

```python
with open('/work/notes.md','a') as fh_:
    fh_.write("""
- Run116-117: insulin given iff a glucose reading at that hour (finger, else lab) >=150; expected rate by tier (finger) .986*1.048,2.069,3.116,3.835; lab .956*1.048,2.005,2.78,3.08; no glucose reading -> no insulin. Glucose read at ~64% of hours.
- Run118: linear decoder from E (slot mean) of [rate at ins bundles, -rbar at glucose bundles, 0 else]: R2 .95 but resid sd .49 (ins) .29 (gread) .19 (other) ~ as big as dev (sd .59) -> TOO NOISY.
- Run119-120: PRELUDE HEAD 3 attends u-token with weight 1.0 (sd 0) at insulin bundles; rate linearly decodable from LN1(u) (prelude norm1) R2 .995 resid .073, ~0 on NE-only/END-only u. At glucose bundles head2->e .93(sd .25), head3->e .63(sd .47).
- DESIGN v2: dims jI (memory), jW (write staging in prelude residual: attn.out row jW reads head3 values; h_jW=0), jR (reference, all-zero content; cancels LN common-mode mu/sd at write (b_in[jI]=c*(e_jW-e_jR)) and at head (with_code_jI - with_code_jR)). LN 1/sd mult noise ~11% on write only.
- NE PROBLEM: injection responding to NE levels double-counts model's severity proxy on factual data (cost >>0.005); titration-noise dev (N - f*N_prev) gives ~no x2 effect. Need off-policy-sensitive design or accept partial NE fix. Insulin first.
""")
import torch.nn.functional as F
def LNt(x, pre): return F.layer_norm(x, (x.shape[-1],), P[pre+'.weight'].float(), P[pre+'.bias'].float())
nb_ = nbA; bd_ = dsA['bd']; prevr = bd_['prev'].values; dts_ = torch.as_tensor(np.asarray(dsA['dts'], dtype=np.float32))
Hprev = torch.zeros(nb_,4,192); Oprev = torch.zeros(nb_,192)
has_p = prevr>=0
Hprev[torch.tensor(has_p)] = HnA[torch.tensor(prevr[has_p])]; Oprev[torch.tensor(has_p)] = OnA[torch.tensor(prevr[has_p])]
dtv_ = dts_.clone(); dtv_[torch.tensor(~has_p)] = 1.0
with torch.no_grad():
    ctxA = ctx_fn(Oprev, Hprev, dtv_, P)
    zA = zpost_mean(ctxA, Ee, U, P)
    hfA = Hprev*torch.exp(-dtv_[:,None,None]*torch.exp(P['core.flow.a'].float()))
    toks = torch.cat([hfA + P['core.slots'].float()[None], Ee[:,None], U[:,None], F.linear(zA, P['z_in.weight'].float(), P['z_in.bias'].float())[:,None]], 1)  # nb x 7 x 192
    tn = LNt(toks, 'core.prelude.0.norm1')
    Wq = P['core.prelude.0.attn.q.weight'].float(); Wkv = P['core.prelude.0.attn.kv.weight'].float()
    Q = tn[:,:4] @ Wq.T; K = tn @ Wkv[:192].T; V = tn @ Wkv[192:].T
    Qh = Q.view(nb_,4,4,48).transpose(1,2); Kh = K.view(nb_,7,4,48).transpose(1,2); Vh = V.view(nb_,7,4,48).transpose(1,2)
    awr = torch.softmax(Qh @ Kh.transpose(-1,-2)/np.sqrt(48), -1)   # nb x 4h x 4q x 7k
print("attn reconstruction max err:", float((awr - AW[:,:,:4,:]).abs().max()))
```