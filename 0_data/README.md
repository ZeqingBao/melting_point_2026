# Combined melting-point data

## Folder organization

The original Bradley processing workflow is preserved in:

- `original_curated/`
- `processed_data/`
- `data_process.ipynb`

The two Bradley CSVs produced by that workflow are copied into `sources/` alongside the other datasets used by the combined-data workflow. `combine_data_process.ipynb` reads only the files in `sources/`.

The five combined-data inputs are:

| Source | Input file | MP input |
|---:|---|---|
| 1 | `sources/1_ChemXplore_ML_tmp_C.csv` | `tmp/ºC` |
| 2 | `sources/2_EGNN_melt_point.csv` | `Melt_C` |
| 3 | `sources/3_PATENTS_ochem_enamine_bradley_begstr_m_training_.parquet` | `Melting Point {measured, converted}` |
| 4 | `sources/test_predictions.csv` | `exp MP` |
| 4 | `sources/train_without_data_augmentation.csv` | `MP` |

## Reproducing the outputs

Open `combine_data_process.ipynb`, restart the kernel, and run all cells in order. The notebook overwrites:

- `combined_data.parquet`
- `multiple_MP_compounds.csv`

The workflow contains no random operations. With unchanged files in `sources/`, unchanged notebook code, and the same software versions, it reproduces the same rows, values, ordering, source lists, and CSV output.

The Parquet dataset will have the same logical contents when compatible software versions are used. Its raw bytes or file hash are not guaranteed to remain identical across different pandas or PyArrow versions because serialization metadata and compression encoding can differ.

The environment used to generate the current files was:

- Python 3.11.13
- pandas 2.3.1
- NumPy 2.3.1
- PyArrow 23.0.0
- RDKit 2025.03.6

The Bradley files under `sources/` are snapshots. Rerunning `data_process.ipynb` changes the files in `processed_data/`, not the copies in `sources/`. Copy newly generated Bradley CSVs into `sources/` before rerunning the combined workflow if those inputs are intentionally updated.

## Processing summary

1. The requested SMILES and MP columns are loaded from each source, and integer source labels are assigned.
2. ChemXplore MP values are parsed from the original `tmp/ºC` field. This preserves negative signs that were lost in some values in `Processed tmp/ºC`. Numeric values with parenthetical uncertainty are retained; censored or approximate entries containing forms such as `<`, `>`, or `≈` are excluded.
3. Numeric patent MP ranges are represented by their midpoint. Other nonnumeric patent values are excluded.
4. SMILES are converted to canonical isomeric SMILES with RDKit. The canonical result must parse successfully a second time.
5. Broad-organic filtering removes metal-containing and multi-component structures. Halogen-containing compounds are retained.
6. Duplicate observation rows with the same canonical SMILES, MP, source label, and source file are removed.
7. Compatible MP observations are resolved with the median and assigned a quality tier.
8. Groups with a total spread above 10 °C are resolved only when a dominant cluster spans no more than 5 °C, contains at least two source labels, and supports at least two-thirds of the observations.
9. Unresolved compounds are omitted from `combined_data.parquet` but retained in the multiple-MP audit.
10. The final consensus MP must be between 0 and 500 °C, inclusive.

Canonicalization retains specified stereochemistry. It does not perform tautomer standardization, neutralization, salt stripping, or stereoisomer merging.

## MP quality tiers

The spread is calculated as the maximum accepted MP minus the minimum accepted MP.

| `MP_quality` | Meaning |
|---|---|
| `single` | One accepted observation from one source label |
| `exact` | Multiple accepted observations with identical numeric MP values |
| `near_exact` | Accepted spread greater than 0 and no more than 1 °C |
| `high_confidence` | Accepted spread greater than 1 and no more than 5 °C |
| `review` | Accepted spread greater than 5 and no more than 10 °C |
| `unresolved` | No acceptable consensus; appears only in the multiple-MP audit |

## `combined_data.parquet`

This is the broad-organic consensus dataset. It includes the review tier so users can select quality levels without rebuilding the data.

| Column | Meaning |
|---|---|
| `SMILES` | Canonical isomeric SMILES. Every value is a valid, round-trippable, single-component, non-metal structure. |
| `MP` | Final consensus melting point in °C. This is the median of the accepted observations. |
| `Source` | Sorted list of unique integer source labels supporting the accepted consensus. Stored as a Parquet `list<int8>`, not as text. |
| `MP_mean` | Arithmetic mean of the accepted MP observations. |
| `MP_std` | Sample standard deviation of accepted MP observations using `ddof=1`. Missing when only one observation was accepted. |
| `MP_mad` | Median absolute deviation of the accepted MP observations from `MP`. |
| `MP_min` | Minimum accepted MP observation. |
| `MP_max` | Maximum accepted MP observation. |
| `MP_range` | Accepted spread: `MP_max - MP_min`. This value determines the quality tier. |
| `N_MP_values` | Number of accepted observation rows after exact same-file duplicates were removed. Identical values from different sources can therefore count separately. |
| `N_distinct_MP` | Number of distinct numeric MP values among the accepted observations. |
| `N_sources` | Number of unique integer source labels supporting the accepted consensus. |
| `MP_quality` | Consensus quality: `single`, `exact`, `near_exact`, `high_confidence`, or `review`. |
| `Consensus_method` | Method used: `single_measurement`, `median_all`, or `dominant_cluster`. |
| `Has_excluded_values` | `True` when one or more observations were excluded while resolving a dominant cluster. |
| `N_excluded_values` | Number of observations excluded from the final consensus. |
| `All_MP_range` | MP spread across all observations before dominant-cluster exclusions. Equal to `MP_range` when no values were excluded. |
| `ContainsHalogen` | `True` when the canonical structure contains F, Cl, Br, I, At, or Ts. Halogens are retained in this dataset. |

`Consensus_method` values mean:

- `single_measurement`: only one observation was available.
- `median_all`: every available observation was accepted and the median was used.
- `dominant_cluster`: the full group spread exceeded 10 °C, but a sufficiently supported cluster spanning no more than 5 °C was accepted.

The current dataset contains 304,683 compounds:

| Quality | Compounds |
|---|---:|
| `single` | 94,901 |
| `exact` | 203,958 |
| `near_exact` | 3,205 |
| `high_confidence` | 2,220 |
| `review` | 399 |

There are 124,021 retained halogen-containing compounds.

## `multiple_MP_compounds.csv`

This is an observation-level audit containing compounds that had more than one distinct numeric MP after structure processing. It contains both resolved and unresolved groups. Compounds with identical MP values across sources are not included because they have no MP disagreement.

| Column | Meaning |
|---|---|
| `SMILES` | Canonical isomeric SMILES shared by the conflicting observations. |
| `MP_reported` | Parsed numeric MP for this observation. ChemXplore signs are corrected, and patent ranges are represented by their midpoint. |
| `MP_raw` | Original MP text from the source before parsing. |
| `Source` | Integer source label for this observation. Unlike the Parquet consensus field, this is one integer per CSV row. |
| `Source_file` | Input filename from which the observation originated. |
| `Accepted_for_consensus` | Whether this observation contributed to the final consensus. All observations are `False` for unresolved groups. |
| `MP_quality` | Quality assigned to the accepted consensus, or `unresolved` when no consensus was accepted. |
| `Consensus_method` | `median_all`, `dominant_cluster`, or `unresolved`. |
| `Group_MP_range` | Maximum minus minimum MP across the complete group before any values were excluded. |

This audit is generated before the final 0–500 °C consensus filter. Consequently, a resolved group can appear in this CSV without appearing in `combined_data.parquet` if its final consensus MP lies outside the requested range.

The current audit contains 7,507 multiple-MP compounds. Of these, 339 remain unresolved and are excluded from the consensus dataset.
