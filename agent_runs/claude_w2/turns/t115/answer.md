`zerr_custom` really is better on every bundle kind:

| bundle kind | zerr_custom | mine |
|---|---|---|
| action | 0.0024 | 0.028 |
| end-only | 0.002 | 0.061 |
| finger glucose | 0.013 | 0.108 |

My glucose-bundle error barely depends on the glucose value or on edge-valued readings, so bin handling is not the cause. Since my E matches the stored Eb, `zerr_custom` must build its encodings differently, probably in how it treats END atoms or empty sets.

Rather than guess, next I decompile `zerr_custom` compactly from its bytecode, one line of opcode arguments per source line, to recover its exact routing logic. That will show the transformation (for example, removing END-atom attributes or a different empty handling), and I can copy it into impl.py.

```python
import dis
def compact(f):
    lines={}; cur=None
    for ins in dis.get_instructions(f):
        if ins.starts_line is not None: cur=ins.starts_line
        if ins.opname in ('RESUME','CACHE','PRECALL','PUSH_NULL','POP_TOP','COPY','SWAP','RETURN_VALUE','NOP'): continue
        a=ins.argrepr if ins.argrepr else ''
        tok={'LOAD_CONST':'c:','LOAD_GLOBAL':'g:','LOAD_FAST':'','LOAD_ATTR':'.','LOAD_METHOD':'.','STORE_FAST':'=>','STORE_SUBSCR':'[]=','BINARY_SUBSCR':'[]','BINARY_OP':'op','COMPARE_OP':'cmp','CALL':'call','KW_NAMES':'kw'}.get(ins.opname,ins.opname[:6]+':')
        lines.setdefault(cur,[]).append(tok+str(a))
    for k_ in sorted(lines): print(k_, ' '.join(lines[k_])[:230])
compact(zerr_custom)
```