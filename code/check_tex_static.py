"""Small static consistency check for the standalone manuscript."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path


path = Path(sys.argv[1] if len(sys.argv) > 1 else "ccop_sisc_complete.tex")
text = path.read_text(encoding="utf-8")
active = "\n".join(
    re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines()
)

labels = re.findall(r"\\label\{([^}]+)\}", active)
references: list[str] = []
for group in re.findall(r"\\(?:eqref|ref|autoref|cref|Cref)\{([^}]+)\}", active):
    references.extend(item.strip() for item in group.split(","))
bibitems = re.findall(r"\\bibitem\{([^}]+)\}", active)
citations: list[str] = []
for group in re.findall(r"\\cite\{([^}]+)\}", active):
    citations.extend(item.strip() for item in group.split(","))

begins = re.findall(r"\\begin\{([^}]+)\}", active)
ends = re.findall(r"\\end\{([^}]+)\}", active)

duplicate_labels = sorted(
    name for name, count in Counter(labels).items() if count > 1
)
missing_labels = sorted(set(references) - set(labels))
missing_citations = sorted(set(citations) - set(bibitems))
environment_delta = Counter(begins)
environment_delta.subtract(ends)
bad_environments = sorted(
    (name, count) for name, count in environment_delta.items() if count
)

figroot_match = re.search(r"\\newcommand\{\\figroot\}\{([^}]+)\}", active)
figroot = figroot_match.group(1) if figroot_match else ""
figure_paths = re.findall(
    r"\\safegraphic(?:\[[^]]*\])?\{([^}]+)\}", active
)
missing_figures: list[str] = []
for raw in figure_paths:
    resolved = raw.replace(r"\figroot", figroot)
    candidate = (path.parent / resolved).resolve()
    if not candidate.is_file():
        missing_figures.append(resolved)

checks = {
    "duplicate_labels": duplicate_labels,
    "missing_labels": missing_labels,
    "missing_citations": missing_citations,
    "missing_figures": missing_figures,
    "environment_delta": bad_environments,
    "inline_math_open": active.count(r"\("),
    "inline_math_close": active.count(r"\)"),
    "display_math_open": len(re.findall(r"(?<!\\)\\\[", active)),
    "display_math_close": len(re.findall(r"(?<!\\)\\\]", active)),
    "brace_delta": active.count("{") - active.count("}"),
}
for name, value in checks.items():
    print(f"{name}: {value}")

failed = any((
    duplicate_labels,
    missing_labels,
    missing_citations,
    missing_figures,
    bad_environments,
    checks["inline_math_open"] != checks["inline_math_close"],
    checks["display_math_open"] != checks["display_math_close"],
    checks["brace_delta"] != 0,
))
raise SystemExit(1 if failed else 0)
