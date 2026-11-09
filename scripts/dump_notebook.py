"""Print the text outputs of an executed notebook (for reviewing results against the narrative)."""
import sys

import nbformat

nb = nbformat.read(sys.argv[1], as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        if o.output_type == "stream":
            txt = o.text
        elif o.output_type in ("execute_result", "display_data"):
            txt = o.get("data", {}).get("text/plain", "<image>" if "image/png" in o.get("data", {}) else "")
        elif o.output_type == "error":
            txt = "ERROR " + o.ename + ": " + o.evalue
        else:
            continue
        if txt.strip():
            print(f"--- cell {i}\n{txt[:1500]}")
