# Source-data integrity ledger

The author repository ignores `data/` and its benchmark driver expects CSV
files with a `target` column. This ledger records only sources compatible with
the paper's Appendix H; it is not a license to substitute a dataset variant.

| Author filename | Paper rows / features | Recovered public identifier | Status |
| --- | ---: | --- | --- |
| physiochemical protein | 45,730 / 9 | OpenML `42903`, target `RMSD` | resolved |
| Food Delivery Time | 45,593 / 10 | Kaggle `rajatkumar30/food-delivery-time`, v1 | resolved; current TabArena variant is not used |
| diamonds | 53,940 / 9 | OpenML `42225`, target `price` | resolved |
| superconductivity | 21,263 / 81 | OpenML `43174`, target `critical_temp` | resolved |
| ailerons | 13,750 / 40 | OpenML `296`, target `goal` | resolved |
| o11 | 5,742 / 1,025 | OpenML `3050` / QSAR-TID-11, target `MEDIAN_PXC50` | resolved |
| miami2016 | 13,932 / 16 | OpenML `43093` / MiamiHousing2016, target `SALE_PRC` | resolved |
| winequality | 6,497 / 12 | OpenML `46964`, target `median_wine_quality` | resolved |

`repro/src/prepare_source_data.py` pins the raw SHA-256 digests and validates
the published shape before writing the source driver's `target`-column CSV
contract. The Food Delivery source is deliberately the original Kaggle v1
file: its 45,593-row/10-predictor shape matches Appendix H, whereas the current
TabArena/OpenML curation does not.

The upstream release provides no preprocessing script, so the only transformations
here are target selection and column naming. This reconstruction is therefore
explicitly checked before any result is used as evidence.
