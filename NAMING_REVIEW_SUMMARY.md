# Naming Review Summary

This document summarizes the comprehensive naming review and improvements made to the building-image-triplet-model project.

## Review Date
2025-10-30

## Overall Assessment

The project **generally follows Python naming conventions (PEP 8)** well:
- ✅ Package naming: `building_image_triplet_model` (snake_case)
- ✅ Class naming: PascalCase (e.g., `GeoTripletNet`, `DatasetProcessor`)
- ✅ Method/function naming: snake_case (e.g., `compute_embeddings`, `train_step`)
- ✅ Variable naming: snake_case (mostly consistent)
- ✅ Constants: UPPER_CASE (where used)

## Issues Identified and Addressed

### 1. Documentation and Clarity (HIGH PRIORITY) ✅ FIXED

#### Problem
- Acronyms (UCB, KNN, HDF5) were not expanded on first use
- "Geo" prefix meaning was ambiguous (Geographic vs Geometric)
- Configuration parameters lacked explanatory comments

#### Solution
**Files Changed:**
- `config.example.yaml` - Added inline comments for all parameters
- `building_image_triplet_model/model.py` - Enhanced docstrings
- `building_image_triplet_model/datamodule.py` - Enhanced docstrings
- `building_image_triplet_model/triplet_dataset.py` - Comprehensive documentation
- `building_image_triplet_model/utils.py` - Module docstring

**Changes Made:**
- Expanded all acronyms on first use:
  - UCB = Upper Confidence Bound
  - KNN = K-Nearest Neighbors
  - HDF5 = Hierarchical Data Format 5
- Clarified "Geo" = "Geographic" (relating to geographic locations/coordinates)
- Added detailed parameter documentation to `__init__` methods
- Added algorithm explanations to key methods
- Enhanced config.example.yaml with purpose of each parameter

### 2. Variable Naming Consistency (MEDIUM PRIORITY) ✅ FIXED

#### Problem
Inconsistent abbreviations for the same concepts:
- `tid`, `tgt`, `target_id` all used for target IDs
- `ds_id_str`, `tgt_id_str`, `sv_id_str` used abbreviated prefixes

#### Solution
**Files Changed:**
- `building_image_triplet_model/triplet_dataset.py`
- `building_image_triplet_model/preprocessing/metadata.py`

**Standardized Naming:**
```python
# Before → After
tid → target_id
tid_to_row → target_id_to_row
anchor_tid → anchor_target_id
valid_tids → valid_target_ids
chosen_tid → chosen_target_id
unique_tids → unique_target_ids
sampled_tids → sampled_target_ids
ds_id_str → dataset_id_str
tgt_id_str → target_id_str
sv_id_str → streetview_id_str
```

**Impact:**
- 15+ variable renamings across 2 files
- Improved code readability
- Consistent naming style throughout
- No breaking changes (all internal variables)

## Issues Intentionally NOT Changed

### 1. Class Names (Considered but Rejected)

**Considered Renaming:**
- `DatasetProcessor` → `BuildingDatasetProcessor`
- `MetadataManager` → `BuildingMetadataManager`
- `EmbeddingComputer` → `FeatureEmbeddingExtractor`

**Decision: NOT CHANGED**

**Rationale:**
- Would require extensive refactoring across multiple files
- Would break backward compatibility if used as library
- Current names are adequate in project context
- Risk/benefit ratio unfavorable for minimal improvement

### 2. Configuration Keys (Not Changed)

**Examples:**
- `ucb_alpha` (kept as-is, added documentation instead)
- `lr` (universally understood in ML)
- `hdf5_path` (standard lowercase for variables)

**Rationale:**
- Changing config keys would break existing user configurations
- Documentation improvements achieved same clarity goal
- Convention in ML community is to use `lr` for learning rate

### 3. DataFrame Column Names (Not Changed)

**Examples:**
- `TargetID`, `DatasetID`, `PatchID` (PascalCase in DataFrames)

**Rationale:**
- Follows pandas/DataFrame conventions
- Distinct from Python variables (intentional differentiation)
- Changing would require extensive data pipeline updates

### 4. HDF5 Casing (Not Changed)

**Current State:**
- Class: `HDF5Writer` (uppercase)
- Variables: `hdf5_path` (lowercase)

**Rationale:**
- This is **CORRECT** per PEP 8
- Acronyms in class names: uppercase (HTTPServer, XMLParser)
- Acronyms in variables: lowercase (http_server, xml_parser)
- Added documentation to clarify HDF5 = Hierarchical Data Format

## Summary of Changes

### Files Modified
1. `config.example.yaml` - Documentation improvements
2. `building_image_triplet_model/model.py` - Docstring enhancements
3. `building_image_triplet_model/datamodule.py` - Docstring enhancements
4. `building_image_triplet_model/triplet_dataset.py` - Docstrings + variable naming
5. `building_image_triplet_model/utils.py` - Module docstring
6. `building_image_triplet_model/preprocessing/metadata.py` - Variable naming

### Total Changes
- **5 files** with documentation improvements
- **2 files** with variable renamings
- **15+ variables** renamed for consistency
- **0 breaking changes** to external APIs

### Testing
✅ All modules import successfully
✅ Python syntax validation passed
✅ No external API changes

## Recommendations for Future

### Short Term
1. Consider adding type hints to more functions (already good coverage)
2. Consider extracting magic numbers to named constants:
   - `0.5` → `DEFAULT_SUCCESS_RATE`
   - `0.3` → `SUCCESS_THRESHOLD`
   - `0.1` → `WARMUP_START_FACTOR`

### Long Term
1. If refactoring for v2.0, consider more specific class names:
   - `DatasetProcessor` → `BuildingDatasetProcessor`
   - But only if doing major version bump

2. Standardize terminology in documentation:
   - Choose: "building images" vs "building typology"
   - Choose: "backbone embeddings" vs "precomputed embeddings"

### Not Recommended
❌ Don't change configuration key names (would break user configs)
❌ Don't change DataFrame column names (would require pipeline changes)
❌ Don't rename classes unless doing major refactoring

## Conclusion

The naming review successfully improved code clarity and consistency while maintaining backward compatibility. The changes focused on:

1. **Documentation** - Making code more accessible to new developers
2. **Consistency** - Eliminating mixed use of abbreviations
3. **Clarity** - Expanding acronyms and adding explanations

All changes were **minimal and surgical**, following the principle of making the smallest necessary improvements without introducing risk or breaking changes.

The codebase now has:
- ✅ Better documented code with expanded acronyms
- ✅ Consistent variable naming (no mixed abbreviations)
- ✅ Clear explanations of key algorithms (UCB, KNN)
- ✅ Enhanced configuration file with inline help
- ✅ No breaking changes

**Result:** Improved maintainability and onboarding experience while preserving stability.
