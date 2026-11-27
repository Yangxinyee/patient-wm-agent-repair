"""Redo the paper's numbers, its generated table rows and Figures 1-5 from the result files in this repository, and compare
each with the copy shipped here (its SHA-256 is in MANIFEST.tsv). CPU only, about a minute.

    pip install -r requirements-figures.txt
    python scripts/check_outputs.py

Runs scripts/paper_numbers.py, paper/make_figures.py and paper/make_teaser.py with this Python, in place, so the shipped
files are overwritten by the redone ones (git checkout paper/ restores them). A PDF records when it was written
(/CreationDate, /ModDate) and Figure 1's carries a document ID made from that time (/ID); these fields are blanked on both
sides before the bytes are compared. Exit status 0 if every output matches.
"""
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = {}
for line in (ROOT / "MANIFEST.tsv").read_text().splitlines():
    if line and not line.startswith(("#", "path\t")):
        path, _, sha = line.split("\t")
        MAN[path] = sha
OUTPUTS = ["paper/numbers.tex"] + sorted(k for k in MAN if k.startswith(("paper/generated/", "paper/figures/")))


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def undated(b: bytes) -> bytes:
    for f in (rb"/CreationDate \(D:[^)]*\)", rb"/ModDate \(D:[^)]*\)", rb"/ID\s*\[<[0-9a-f]*><[0-9a-f]*>\]"):
        b = re.sub(f, b"", b)
    return b


shipped = {k: (ROOT / k).read_bytes() for k in OUTPUTS}
changed = [k for k in OUTPUTS if sha(shipped[k]) != MAN[k]]
if changed:
    sys.exit("these files differ from MANIFEST.tsv before the check; restore them first (git checkout paper/): "
             + ", ".join(changed))
for cmd in ("scripts/paper_numbers.py", "paper/make_figures.py", "paper/make_teaser.py"):
    print(f"$ python {cmd}", flush=True)
    subprocess.run([sys.executable, cmd], cwd=ROOT, check=True)
bad = 0
for k in OUTPUTS:
    new = (ROOT / k).read_bytes()
    if sha(new) == MAN[k]:
        verdict = "identical"
    elif k.endswith(".pdf") and undated(new) == undated(shipped[k]):
        verdict = "identical but for the time fields"
    else:
        verdict, bad = "DIFFERS", bad + 1
    print(f"{verdict:32s} {k}")
print(f"{len(OUTPUTS) - bad} of {len(OUTPUTS)} outputs match the shipped files")
sys.exit(1 if bad else 0)
