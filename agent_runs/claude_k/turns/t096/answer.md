There is no input history. The notebook (4.9 KB) is all I wrote down, so runs 74–86 are lost except for whatever variables they left behind.

To design an edit I first need the model's computation in detail again. My notes cover only fragments: GELU MLPs, the z_post input order, the value head layout. So I'll reread /model/ARCHITECTURE.md, which is held in the variable `arch`, in chunks.

Success means the update path is clear: how u_k, the 4 slots of h, out, z and the view feed the value head. In particular I need to know whether a dedicated "dose memory" channel could be added to the state and read in the head. If anything contradicts my notes, I'll correct the notes.

```python
print(len(arch))
print(arch[:7000])
```