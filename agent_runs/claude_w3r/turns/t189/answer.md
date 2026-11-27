The architecture spec settles several points:
- The recurrent state h is the post-coda y.
- `out` = out_norm(mean_slots(y)).
- The heads read `view = out + out_norm(decayed h mean) + gap + z_head(z)`. The value head then adds a last-value register and `embed_transition[code*17+last bin]`.
- u (the action encoding) enters the core only through the prelude tokens, then E, then the blocks and b_in.

So my plan is a dose channel that boosts the clean u signal into the state and reads it at the glucose and MAP value heads. Because the black-box step functions use `Wt`, I can simulate weight edits locally.

Before going further I'm saving these facts and the plan to the notebook, since my notes had lost runs 118–170. Then I'll check what `step_at` returns.

```python
add = """
## Session 7 (restart 7) - recovered state (runs 171-176)
- Runs 118-170 were NOT noted. Recovered: kernel fns (no source) use dict Wt (torch f64 weights) -> can simulate edits by
  temporarily modifying Wt[...]. Head fns use W (numpy). step_at(B,rows,EU,k,du,de)->core_step_v; stay_data(s)->(B,rows,EU);
  SX(5301,992)=fwd_2 states (SST stay, SHR hour). UX(9918,192)=u vectors (YXi/YXn rates); run170: linear probe of
  ins/ne rate from u holds under all 4 edits (slope~1, resid sd .03-.17) incl OOD ne x2.
- ARCH (re-read): new h = y = coda(c_out(h_mid)), h_mid = abar*h + sp*b_in(E) + (blocks(h+E)-(h+E)); out=out_norm(mean y).
  view = out + out_norm(mean_slots(h after time flow)) + gap + z_head(z). with_code = q + code emb + last_numeric[has,log(1+age)/10,z_last,z_last_rate]
  + bins.value[last bin] + bins.rate[last rate bin] + last_status. value logits = value_bin(with_code) + embed_transition[code*17+lastbin].
  Dose edit doubles rate&dose of EVERY insulin atom from h12 (incl END atoms). Readings drawn & fed back from h13.
- Timing: reading bundle s:00 (gf,map), action bundle s:05 (starts->u, END of s-1 infusion->e). Reading s+1:00 is predicted from
  state after s:05 bundle -> lag1 (r_s in u) and lag2 (r_{s-1} in END atom e) are both in the action bundle.
- PLAN A (3 stages): (1) prelude attn.out rank-1 per drug from head2 value (head2 attends u 100%), into a LOW-VARIANCE slot-token
  direction d (zero at no-action: v.V2(norm1(0))=0); (2) core.b_in rank-1 per drug reading that signal from E, writing direction a
  into h_mid; (3) value head dead units gated on code (gf / map_arterial) reading the channel, tilt vector -> mean shift.
  Compensation: insulin -b1*m(last gf bin) via embed_transition rows; norepi intercept. Calibrate gains end-to-end with dose_edit.
"""
open('/work/notes.md','a').write(add)
s0=int(SST[0]); print('first stay in SX', s0, 'n stays', len(set(SST.tolist())))
B_,rows_,EU_=stay_data(s0)
print(type(B_), len(B_), type(rows_), np.shape(rows_), type(EU_), len(EU_))
print('B[0..3]', [ (b[0], [(a if not hasattr(a,'__len__') else a) for a in b[1]][:3]) for b in B_[:4]])
r=step_at(B_,rows_,EU_,5)
print(type(r), len(r) if hasattr(r,'__len__') else '')
for i,x in enumerate(r if isinstance(r,(tuple,list)) else [r]):
    print(i, type(x).__name__, tuple(x.shape) if hasattr(x,'shape') else (x if not hasattr(x,'__len__') else len(x)))
```