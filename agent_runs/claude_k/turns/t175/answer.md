The raw records are clean: insulin dose = rate exactly, and norepinephrine dose = 4.8 × rate exactly for every stay. So weight doesn't vary and the recorded attributes are simple. The model adds END atoms itself when an infusion finishes, but the earlier split showed those are not the main source of error.

The rate-dependent error must therefore be in how my rebuild embeds a drug atom: which categorical/unit tables, which event-kind row, or how bins and z-scores enter. I'll list the embedding and action-encoder tensors. Then I'll try to rebuild the two atoms of the first bundle of stay 1010001440 (insulin 1.5, norepinephrine 0.30, no END atoms) term by term.

If `drug_atoms` reproduces my hand-built atom under the documented conventions, the bug is downstream (the set encoder). If not, the mismatch shows which embedding term is wrong.

```python
print([ (n_,SHAPES[n_]) for n_ in NAMES if n_.startswith('embed') or n_.startswith('encode_action')])
print('event_kinds',VOCAB['event_kinds'],'atom_kinds',VOCAB['atom_kinds'])
print('codes 8-12', VOCAB['codes'][8:13])
V0=np.asarray(V_[0]); V1=np.asarray(V_[1])
base_=np.asarray(atom_vec(8)); print('|V0|',np.linalg.norm(V0),'|V0-atom_vec(8)|',np.linalg.norm(V0-base_))
```