The listing shows how the core is built. The prelude, the two recurrent layers and the coda are all pre-norm transformer layers with residual connections and attention without biases. The prelude has no cross-attention. The dose can therefore reach the four slots, and so the state the heads read, only through prelude attention from the u_k token (or the z token). Feed-forward layers act on each token separately and cannot move it there.

Next I will check what my local re-implementation (class M in impl.py) already covers. If it runs a full core step (prelude, blocks, coda, out), I can test locally how a dose signal injected into u_k propagates. If it stops at the encoders or z_post, I will measure the propagation with a "states" forward pass instead.

```python
import ast
tree=ast.parse(src)
for node in tree.body:
    if isinstance(node, ast.ClassDef):
        print('class', node.name, [f.name for f in node.body if isinstance(f, ast.FunctionDef)])
lines=src.splitlines()
for i,l in enumerate(lines):
    if 'def ' in l and 'class' not in l: print(i, l.strip()[:110])
print('--- tail of impl.py ---')
print('\n'.join(lines[-40:]))
```