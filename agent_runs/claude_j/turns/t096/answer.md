The source of the replica functions cannot be recovered because they were defined in the kernel without being saved to files. The functions and their results still exist, but I cannot read their logic. This time I will write the replica as source files in `/work` (for example `/work/replica.py`) and exec them, so the code survives restarts. I will also write progress into the notebook straight away.

Before rebuilding, I want to see how far the old replica got: the function signatures, the fwd_3 summary (which stays and bundles), and validation scalars such as `o_chk` and `z_err*`. I also want the structure of ARCHITECTURE.md, so I can re-read only the sections I need.
- **Old replica usable:** if it reproduced `out` and z to about 1e-6, I can call the old functions while I re-implement them cleanly.
- **Otherwise:** I will rebuild from the document alone.

```python
for fnm_ in ['build_bundles','bundle_vecs','encode','encode_v','enc_e','step','step0','step0v','mha','mlp','tlayer','block','ffn','ctx_of','drug_atom','meas_atom','meas_atom_gl','visit_atom','drug_end','one_step_err','per_test_err','run_variant3','true_shift','rows_truth','z_err4','a_err2','err_at']:
    try:
        f_ = g[fnm_]
        print(fnm_, str(inspect.signature(f_))[:150], '| calls:', [n for n in f_.__code__.co_names][:25])
    except Exception as ex_:
        print(fnm_, 'ERR', ex_)
print(open('/work/fwd_3/summary.json').read()[:1500])
for k_ in ['o_chk','Es_n','r0_','rp','row']:
    print(k_, g.get(k_))
print('ARCH chars', len(arch))
for i_, line_ in enumerate(arch.splitlines()):
    if line_.startswith('#'):
        print(i_, line_)
```