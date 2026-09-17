# Combined melting-point data

![Combined melting-point data workflow](data_combining_process_diagram.png)

The diagram summarizes the core combination and MP-consensus workflow before the
final molecular-weight filter. Its 304,683-compound count is the pre-MW consensus
count. The final saved dataset contains 304,509 compounds after applying the
strict `MW < 1000` filter described below.

## Folder organization

The original Bradley processing workflow is preserved in:

- `original_curated/`
- `processed_data/`
- `data_process.ipynb`

The two Bradley CSVs produced by that workflow are copied into `sources/` alongside the other datasets used by the combined-data workflow. `combine_data_process.ipynb` reads only the files in `sources/`.

`source_distribution_overlap_EDA.ipynb` independently rebuilds the cleaned
source-level observations for distribution and overlap visualizations. It does
not generate or modify the combined dataset.

The five combined-data inputs are:

| Source | Input file | MP input |
|---:|---|---|
| 1 | `sources/1_ChemXplore_ML_tmp_C.csv` | `tmp/ºC` |
| 2 | `sources/2_EGNN_melt_point.csv` | `Melt_C` |
| 3 | `sources/3_PATENTS_ochem_enamine_bradley_begstr_m_training_.parquet` | `Melting Point {measured, converted}` |
| 4 | `sources/test_predictions.csv` | `exp MP` |
| 4 | `sources/train_without_data_augmentation.csv` | `MP` |

### Source publications

| Source | Publication | DOI |
|---:|---|---|
| 1 | *Machine Learning Pipeline for Molecular Property Prediction Using ChemXploreML* | [10.1021/acs.jcim.5c00516](https://doi.org/10.1021/acs.jcim.5c00516) |
| 2 | *Multi-Conformation Enhanced Equivariant Graph Neural Network: Advancing Melting Point Prediction Accuracy for Organic Small Molecules* | [10.1021/acsomega.5c04588](https://doi.org/10.1021/acsomega.5c04588) |
| 3 | *The Development of Models to Predict Melting and Pyrolysis Point Data Associated with Several Hundred Thousand Compounds Mined from PATENTS* | [10.1186/s13321-016-0113-y](https://doi.org/10.1186/s13321-016-0113-y) |
| 4 | *Prediction of Melting Points of Chemicals with a Data Augmentation-Based Neural Network Approach* | [10.1021/acsomega.5c00205](https://doi.org/10.1021/acsomega.5c00205) |

## Reproducing the outputs

Open `combine_data_process.ipynb`, restart the kernel, and run all cells in order. The notebook overwrites:

- `combined_data.parquet`
- `multiple_MP_compounds.csv`

The notebook first writes the resolved consensus output and then applies the
standalone molecular-weight cell to `combined_data.parquet`. That cell
recalculates RDKit average molecular weight, adds the `MW` column, retains only
`MW < 1000`, and atomically replaces the Parquet file. It is important to run
this cell after the main Parquet-writing cell; rerunning the earlier writing cell
alone recreates the pre-MW output.

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
2. ChemXplore MP values are parsed from the original `tmp/ºC` field. This preserves negative signs that were lost in some values in `Processed tmp/ºC`. For a numeric value with parenthetical uncertainty, the central numeric value is retained and the parenthetical part is not propagated. Censored or approximate entries containing forms such as `<`, `>`, or `≈` are excluded.
3. Strict numeric PATENTS MP ranges are represented by their midpoint. For example, `92–93` becomes one processed observation with `MP_reported = 92.5`. Other nonnumeric PATENTS values are excluded. The original text remains available as `MP_raw` in the multiple-MP audit when that observation belongs to an audited group.
4. SMILES are converted to canonical isomeric SMILES with RDKit. The canonical result must parse successfully a second time.
5. Broad-organic filtering removes metal-containing and multi-component structures. Halogen-containing compounds are retained.
6. Duplicate observation rows with the same canonical SMILES, MP, source label, and source file are removed.
7. Compatible MP observations are resolved with the median and assigned a quality tier.
8. Groups with a total spread above 10 °C are resolved only when a dominant cluster spans no more than 5 °C, contains at least two source labels, and supports at least two-thirds of the observations.
9. Unresolved compounds are omitted from `combined_data.parquet` but retained in the multiple-MP audit.
10. The final consensus MP must be between 0 and 500 °C, inclusive.
11. RDKit `Descriptors.MolWt` calculates average molecular weight from each saved canonical SMILES. The final Parquet retains only compounds with `MW < 1000` and includes `MW` as a column. This filter is applied after MP consensus and the 0–500 °C consensus filter.

Canonicalization retains specified stereochemistry. It does not perform tautomer standardization, neutralization, salt stripping, or stereoisomer merging.

“Multi-component” means that RDKit `Chem.GetMolFrags` identifies anything
other than one molecular fragment, commonly represented by a dot in SMILES.
Metal filtering uses the explicit `METAL_SYMBOLS`/atomic-number set defined in
the notebook. Halogens are not classified as metals and are retained.

Duplicate detection, exact agreement, spread calculations, medians, and quality
tiers use the parsed, unrounded numeric MP values. Thus `100` and `100.0`
are numerically identical, while nearby values are not made identical by
rounding.

For PATENTS ranges, only the midpoint becomes an MP observation. The range
endpoints are not used to calculate `MP_mean`, `MP_std`, `MP_mad`,
`MP_min`, `MP_max`, or `MP_range`. Those statistics are calculated
afterward from the accepted processed observations for a canonical compound.

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

This is the broad-organic, molecular-weight-filtered consensus dataset. It includes the review tier so users can select quality levels without rebuilding the data. All rows have a consensus MP from 0 to 500 °C, inclusive, and `MW < 1000`.

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
| `MW` | RDKit average molecular weight from `Descriptors.MolWt`, calculated from the canonical isomeric SMILES. Only values strictly below 1,000 Da are retained. This is not monoisotopic exact mass. |

`Consensus_method` values mean:

- `single_measurement`: only one observation was available.
- `median_all`: every available observation was accepted and the median was used.
- `dominant_cluster`: the full group spread exceeded 10 °C, but a sufficiently supported cluster spanning no more than 5 °C was accepted.

When more than one candidate dominant cluster exists, candidates are prioritized
by: (1) the greatest number of distinct source labels, (2) the greatest number
of observations, and (3) the smallest spread. The selected cluster must still
contain at least two observations from at least two sources and represent at
least two-thirds of the complete observation group.

The current saved dataset contains 304,509 compounds after the strict
`MW < 1000` filter:

| Quality | Compounds |
|---|---:|
| `single` | 94,762 |
| `exact` | 203,925 |
| `near_exact` | 3,204 |
| `high_confidence` | 2,219 |
| `review` | 399 |

The MW filter removed 174 compounds from the 304,683-compound pre-MW consensus
output. The retained MW range is 18.015–997.694 Da. There are 123,975 retained
halogen-containing compounds. No separate file containing every MW-excluded
compound is written.

## `multiple_MP_compounds.csv`

This is an observation-level audit containing compounds that had more than one distinct numeric MP after structure processing. It contains both resolved and unresolved groups. Compounds with identical MP values across sources are not included because they have no MP disagreement.

| Column | Meaning |
|---|---|
| `SMILES` | Canonical isomeric SMILES shared by the conflicting observations. |
| `MP_reported` | Parsed numeric MP for this observation. ChemXplore signs are corrected, and PATENTS ranges are represented by their midpoint. |
| `MP_raw` | Original MP text from the source before parsing. |
| `Source` | Integer source label for this observation. Unlike the Parquet consensus field, this is one integer per CSV row. |
| `Source_file` | Input filename from which the observation originated. |
| `Accepted_for_consensus` | Whether this observation contributed to the final consensus. All observations are `False` for unresolved groups. |
| `MP_quality` | Quality assigned to the accepted consensus, or `unresolved` when no consensus was accepted. |
| `Consensus_method` | `median_all`, `dominant_cluster`, or `unresolved`. |
| `Group_MP_range` | Maximum minus minimum MP across the complete group before any values were excluded. |

This audit is generated before both the final 0–500 °C consensus filter and the
final `MW < 1000` filter. Consequently, a resolved group can appear in this
CSV without appearing in `combined_data.parquet` if its final consensus MP
lies outside the requested range or its molecular weight is at least 1,000 Da.
The audit is not a complete log of all MW exclusions because it contains only
compounds with more than one distinct numeric MP.

The current audit contains 7,507 multiple-MP compounds. Of these, 339 remain unresolved and are excluded from the consensus dataset.
