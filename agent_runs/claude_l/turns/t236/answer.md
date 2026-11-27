The kernel is intact: 137 helper functions and the exact-replica machinery (bstep, rollout, z_post parts) are still loaded. The only saved edit is z_head.weight set to zero, which costs nothing in factual fit. So the Stage-1 "residual dose through the z token" edit has not been saved yet, and the probe edits are not active.

Run 217 used `eval_dir`, and the injection code is in /work/inj.py (`bstep_z`, `info_of`, `tf_stay_z`). Before building anything I need to re-read that code so I don't redo or contradict runs 177–217. I'll print inj.py and the source of the few kernel functions behind the last analysis, truncated.

What would confirm my picture: `bstep_z` lets z_post/z_in be replaced by a custom z-token function, and `eval_dir` scores h-directions for a lasting z-loop dose memory.

```python
import inspect
print(open('/work/inj.py').read()[:2800])
for fn in ['eval_dir', 'make_zfn', 'try_combo', 'true_shift']:
    try:
        s = inspect.getsource(globals()[fn]); print('----', fn, len(s)); print(s[:1100])
    except Exception as ex:
        print('----', fn, 'no source', type(ex).__name__)
```