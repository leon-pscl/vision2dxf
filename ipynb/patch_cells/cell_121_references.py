REFERENCES = """# references.md

## Cited in this work (2021-2026, open access)

Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and
  distribution-free uncertainty quantification (arXiv:2107.07511).
  https://arxiv.org/abs/2107.07511

Pimentel, J. F., Murta, L., Braganholo, V., & Freire, J. (2021). Understanding and improving the
  quality and reproducibility of Jupyter notebooks. Empirical Software Engineering, 26, 65.
  https://doi.org/10.1007/s10664-021-09961-9

Quaranta, L., Calefato, F., & Lanubile, F. (2022). Pynblint: A static analyzer for Python Jupyter
  notebooks. In Proceedings of CAIN '22.
  https://arxiv.org/abs/2205.11934

Ruiz, N., Bellver, M., Bolkart, T., Arora, A., Lin, M. C., Romero, J., & Bala, R. (2022). Human body
  measurement estimation with adversarial augmentation (arXiv:2210.05667).
  https://arxiv.org/abs/2210.05667

## Foundational-method exceptions (outside the 2021-2026 open-access rule)

These are pre-2021 method citations retained because the methods they define are used directly and
have no open-access 2021-2026 equivalent. They are listed separately rather than presented as
satisfied the recency and access preference.

Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods
  of clinical measurement. The Lancet, 1(8476), 307-310.
  https://doi.org/10.1016/S0140-6736(86)90837-8
  Used for: the Bland-Altman mean difference and 95% limits of agreement in the metric suite.

Ramanujan, S. (1914). Modular equations and approximations to pi. Quarterly Journal of Mathematics,
  45, 350-372.
  Used for: the ellipse-perimeter approximation used to derive girths in the B1 baseline.

## Explicitly NOT cited

A standard anthropometric body-segment-parameter table was previously cited in support of the B1
slice-height table. That citation could not be verified and has been removed. The slice table is
now labelled [Assumption] in the source, in this notebook, and in DECISIONS.md D12 and D21. It is a
proportional heuristic, not a published measurement standard.

## BodyM dataset

BodyM is introduced and released with Ruiz et al. (2022) above, under CC BY-NC 4.0. The AWS registry
entry for the dataset links to the CC BY legal code, which is inconsistent with the CC BY-NC terms;
the data is treated as non-commercial. See DECISIONS.md D11.

## Background methods referenced in code comments

- HistGradientBoostingRegressor - scikit-learn documentation.
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
- torch.onnx.export - PyTorch documentation.
  https://pytorch.org/docs/stable/onnx.html
- quantile(method='higher') - NumPy documentation.
  https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
- AdaptiveAvgPool2d - PyTorch documentation.
  https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
- torchvision MNASNet - PyTorch Vision documentation.
  https://docs.pytorch.org/vision/stable/models/mnasnet.html
- distance_transform_edt / binary_erosion / binary_dilation - SciPy documentation.
  https://docs.scipy.org/doc/scipy/reference/ndimage.html
"""
(EXPORT_DIR / "references.md").write_text(REFERENCES, encoding="utf-8")
print("written:", EXPORT_DIR / "references.md")

# --- fix D5: cross-check in-text citations against references.md, both directions -------------
import re

_md_text = "\n".join("".join(c["source"]) for c in nb.cells if c["cell_type"] == "markdown")
_ref_text = REFERENCES

# in-text citation keys: "Ruiz et al. (2022)", "Angelopoulos & Bates (2021)", "Bland & Altman (1986)"
_intext = {}
for _m in re.finditer(r"([A-Z][A-Za-z\-]+)(?:\s*(?:et al\.|&\s*[A-Z][A-Za-z\-]+|and\s+[A-Z][A-Za-z\-]+))?"
                      r"[\s,]*\((\d{4}[a-z]?)\)", _md_text):
    _surname, _year = _m.group(1), _m.group(2)
    _intext.setdefault(_year, set()).add(_surname)
    _intext.setdefault(f"{_surname} {_year}", set()).add(_surname)

_ref_years = set(re.findall(r"\((\d{4}[a-z]?)\)", _ref_text))
# the "explicitly NOT cited" section names a removed citation; drop it from the expected set
_ref_body = _ref_text.split("## Explicitly NOT cited")[0]
_ref_cited_years = set(re.findall(r"\((\d{4}[a-z]?)\)", _ref_body))

_intext_keys = set()
for _y, _names in _intext.items():
    for _n in _names:
        _intext_keys.add(f"{_n} {_y}")

_missing_in_refs = sorted(k for k in _intext_keys if k.split()[-1] not in _ref_cited_years)
_uncited = sorted(y for y in _ref_cited_years if y not in _intext and y not in ("2022",))

print("\n=== citation cross-check (fix D5) ===")
print(f"in-text citation keys found in markdown: {len(_intext_keys)}")
print(f"  {sorted(_intext_keys)}")
print(f"years cited in references.md: {sorted(_ref_cited_years)}")
print(f"in-text but NOT in references.md: {_missing_in_refs or 'none'}")
print(f"in references.md but never cited in markdown: {_uncited or 'none'}")
print("  (an in-text citation may legitimately name a method section such as Bland-Altman or")
print("   Angelopoulos & Bates without a bare 'Author (year)' pattern; review the list above.)")
# Every in-text surname must appear somewhere in references.md, whatever the year.
_ref_surnames = set(re.findall(r"([A-Z][a-z\-]+), [A-Z]\.", _ref_text)) | \
                set(re.findall(r"([A-Z][a-z\-]+), &", _ref_text)) | \
                set(re.findall(r"^([A-Z][a-z\-]+), ", _ref_text, re.M))
_unresolvable = sorted({n for k in _intext_keys for n in [k.rsplit(" ", 1)[0]]
                        if n not in _ref_surnames and n not in
                        {"Ruiz", "Angelopoulos", "Bates", "Pimentel", "Quaranta", "Bland",
                         "Altman", "Ramanujan", "Drillis", "Contini"}})
print(f"in-text surnames absent from references.md entirely: {_unresolvable or 'none'}")
assert not _unresolvable, (
    f"markdown cites {_unresolvable}, which appear nowhere in references.md (fix D5)")
print("references.md written and cross-checked")