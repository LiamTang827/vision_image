# Assignment Submission Checklist

## Before You Start

- [ ] Dataset downloaded and extracted
  - [ ] `gallery/gallery/` folder contains 28,493 images
  - [ ] `gallery/query/` folder contains 50 query images
  - [ ] `query_txt/` folder contains 50 .txt files with bounding boxes

- [ ] Python 3.7+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)

## Testing Phase

- [ ] Run `python test_setup.py` - all checks pass
- [ ] Run `python quick_test.py` - first 5 queries complete successfully
- [ ] Check generated test images look reasonable

## Full Execution

- [ ] Run `python instance_search.py`
- [ ] Wait for completion (1-30 hours depending on hardware)
- [ ] No errors during execution

## Verify Output Files

### Rank Lists (Required)
- [ ] `rankList_ResNet50.txt` exists
  - [ ] Contains exactly 50 lines
  - [ ] Each line starts with "Q1:", "Q2:", ..., "Q50:"
  - [ ] Each line contains 28,493 space-separated numbers
  - [ ] Numbers are integers from 0 to 28492

- [ ] `rankList_SIFT_Histogram.txt` exists
  - [ ] Contains exactly 50 lines
  - [ ] Format matches above requirements

### Visualizations (Required for Queries 1-5)
- [ ] `results_query_0_ResNet50.png` exists
- [ ] `results_query_1_ResNet50.png` exists
- [ ] `results_query_2_ResNet50.png` exists
- [ ] `results_query_3_ResNet50.png` exists
- [ ] `results_query_4_ResNet50.png` exists

### Report (Required)
- [ ] `REPORT.md` reviewed and edited with your name/ID
- [ ] `REPORT.pdf` created (run `python convert_report.py` or convert manually)
- [ ] Report is 1-6 pages
- [ ] Report includes:
  - [ ] Introduction
  - [ ] Methodology for both methods
  - [ ] Implementation details
  - [ ] Results (reference to visualization images)
  - [ ] Analysis and discussion
  - [ ] Conclusion
  - [ ] References

### Code (Required)
- [ ] `instance_search.py` works correctly
- [ ] Code is well-commented
- [ ] No hardcoded absolute paths

## Final Review

### Method 1 (ResNet50) Requirements
- [ ] ✓ Deep neural network-based (ResNet50 CNN)
- [ ] ✓ Extracts features from ROIs defined by bounding boxes
- [ ] ✓ Computes similarity and ranks all 28,493 images
- [ ] ✓ Generates complete rank list

### Method 2 (SIFT+Histogram) Requirements
- [ ] ✓ Traditional computer vision approach
- [ ] ✓ Different from Method 1
- [ ] ✓ Properly implemented and functional
- [ ] ✓ Generates complete rank list

### Visualization Requirements
- [ ] ✓ Shows top-10 results for at least one method
- [ ] ✓ Covers queries 1-5 (files 0-4)
- [ ] ✓ Query image shows bounding boxes
- [ ] ✓ Retrieved images are displayed

### Overall Requirements
- [ ] ✓ Two distinct methods implemented
- [ ] ✓ At least one method is deep learning-based
- [ ] ✓ Both methods work correctly
- [ ] ✓ Bounding boxes used for object localization
- [ ] ✓ Complete rank lists for all 50 queries
- [ ] ✓ Visualizations included
- [ ] ✓ Report describes methods and analysis

## Prepare Submission Package

### Create Submission ZIP
```bash
zip -r StudentName_StudentID_InstanceSearch.zip \
    instance_search.py \
    REPORT.pdf \
    rankList_ResNet50.txt \
    rankList_SIFT_Histogram.txt \
    results_query_0_ResNet50.png \
    results_query_1_ResNet50.png \
    results_query_2_ResNet50.png \
    results_query_3_ResNet50.png \
    results_query_4_ResNet50.png \
    README.md
```

### ZIP Contents Checklist
- [ ] `instance_search.py` (main code)
- [ ] `REPORT.pdf` (1-6 pages)
- [ ] `rankList_ResNet50.txt` (Method 1 rankings)
- [ ] `rankList_SIFT_Histogram.txt` (Method 2 rankings)
- [ ] `results_query_0_ResNet50.png` (Query 1 visualization)
- [ ] `results_query_1_ResNet50.png` (Query 2 visualization)
- [ ] `results_query_2_ResNet50.png` (Query 3 visualization)
- [ ] `results_query_3_ResNet50.png` (Query 4 visualization)
- [ ] `results_query_4_ResNet50.png` (Query 5 visualization)
- [ ] `README.md` (optional, but helpful)

### File Size Check
- [ ] ZIP file is reasonable size (< 50 MB)
- [ ] If too large, images may need compression

## Before Submission

- [ ] Student name and ID in REPORT.pdf
- [ ] All file names are correct (no typos)
- [ ] ZIP file opens correctly
- [ ] Test: Extract ZIP to new folder and verify contents
- [ ] Submission deadline noted: _______________
- [ ] Canvas submission portal accessible

## Submission

- [ ] Upload ZIP file to Canvas
- [ ] Verify upload successful
- [ ] Download uploaded file to confirm
- [ ] Keep backup copy of submission
- [ ] Screenshot of successful submission

## Post-Submission

- [ ] Confirmation email received (if applicable)
- [ ] Mark submission date/time: _______________
- [ ] Note: Late submissions may have penalties

---

## Point Distribution (60% Total)

### Implementation (40-60%)
- [ ] Two methods implemented correctly (40% if one works, 60% if both work)
- [ ] Deep learning method included (+bonus)
- [ ] Object localization with bounding boxes (+bonus)

### Report (20%)
- [ ] Clear methodology description
- [ ] Implementation details
- [ ] Results analysis
- [ ] Discussion and comparison

### Rank Lists (20%)
- [ ] Correct format
- [ ] Complete rankings (all 28,493 images)
- [ ] All 50 queries processed
- [ ] Retrieval performance

---

## Emergency Contacts

- **Teaching Assistant**: Renwei Yang
- **Email**: renweyang2-c@my.cityu.edu.hk
- **Office Hours**: _______________

---

## Notes

- Save this checklist and check items as you complete them
- Keep all original files as backup
- Don't wait until the last day
- Test on a fresh system if possible
- Ask for help early if stuck

**Good luck! 🍀**
