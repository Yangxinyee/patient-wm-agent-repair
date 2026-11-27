The kernel still holds the helpers. There is an exact core step (`bstep`), a teacher-forced stay (`tf_stay`), an offline rollout (`rollout`), and an injection variant (`bstep_z`, `tf_stay_z` in inj.py), which was probably the z-token mechanism I was building. I need their exact interfaces before building the edit, so I won't write new code that duplicates what exists. I'll print the injection and teacher-forcing code and the signatures of the rollout and step functions. If inj.py already does the copy of u plus residual injection, I can go straight to fitting the residual decoders and calibrating the gain.

```python
import inspect
print(open('/work/inj.py').read())
print('-----tf.py')
print(open('/work/tf.py').read())
for f in ['wm2.py','wm2_rollout.py']:
    src=open('/work/'+f).read()
    print('-----',f, [l for l in src.splitlines() if l.startswith('def ')])
```