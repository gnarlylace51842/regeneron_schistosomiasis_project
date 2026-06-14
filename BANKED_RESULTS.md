# Banked contribution (floor) — "Localize, don't classify"

*Self-contained summary of the current, defensible result. This is the submission floor:
if the planned method work does not pan out, this stands on its own. If it does, this
becomes the motivation (problem + mechanism) for the method.*

Date frozen: 2026-06-13.

---

## Abstract

Automated microscopy diagnosis is usually framed as **image classification** (is this
sample positive?). We show that this framing is brittle to the **distribution shift that
occurs between field studies / sites**, and that re-framing the task as **object
detection** (localize the pathogen, then aggregate) is robust to the same shift — across
two independent diseases.

On schistosomiasis brightfield microscopy from two field studies (mar2020 → nov2021), an
ImageNet-pretrained classifier learns the source study well (within-study AUC ≈ 0.85) but
**collapses to chance on the unseen study (AUC ≈ 0.52, 95% CI through 0.50)**, while a
YOLOv8 egg **detector** evaluated on the *same patients* holds at **AUC 0.914
[0.859, 0.961]** (reproducing the established cross-study detection result of de Leon
Derby et al., 2025 — a reproduction, not a new result; the contribution is the *contrast*).
Saliency analysis shows the mechanism: the classifier attends to global
background / illumination rather than the egg. On a second, independent dataset
(Chula-ParasiteEgg-11, 11 species), a single-class egg detector trained on 8 species
**generalizes to 3 species it never saw (mAP50 0.993 vs in-distribution 0.992)** — a
class-agnostic detector handles novel pathogen categories that a closed-set classifier
structurally cannot. Together: **localizing the invariant target, rather than classifying
the global image, confers robustness to both site shift and novel-species shift.**

---

## The finding, in four results

**1. Classification collapses across the study shift; detection does not (same direction, same patients).**

| Model | within-study (mar val) | cross-study (nov, zero-shot) |
|---|---|---|
| MobileNetV2 | 0.854 | **0.517 [0.432, 0.600]** |
| EfficientNet-B0 | 0.857 | **0.504 [0.426, 0.590]** |
| **YOLOv8s detector** | — | **0.914 [0.859, 0.961]** |

380 nov2021 test patients (61 positive / 319 negative). Detector operating points:
sensitivity 0.738 at 96.5% specificity (M&E TPP), 0.557 at 99.5% specificity (TI&S TPP).
For reference, de Leon Derby et al. (2025) report **76%** BF sensitivity at the same M&E target
on this exact March→Nov split — so our detector **reproduces (slightly under) their established
result**; it is not claimed to beat it. (They report no patient-level AUC.) Classifier and
detector CIs do not overlap.

**2. Mechanism.** On nov2021 images that *contain* eggs, the cross-study classifier outputs
p(positive) ≈ 0.04–0.16 (it misses them) and Grad-CAM shows its attention on diffuse
background/illumination, not the egg (~8 px in a 224-px view).

**3. Cross-species generalization (2nd disease, Chula-ParasiteEgg-11).** Single-class
`parasite_egg` YOLOv8s, trained on 8 species, evaluated on 3 held-out species
(Ascaris lumbricoides, Capillaria philippinensis, Trichuris trichiura):

| | in-distribution (8 seen) | out-of-distribution (3 unseen) |
|---|---|---|
| mAP50 | 0.992 | **0.993** |
| mAP50-95 | 0.943 | 0.896 |

**4. Conceptual.** A class-agnostic detector localizes unseen pathogen species; a closed-set
classifier has no output for a category it never trained on. Detection is the structurally
correct choice for open-set / novel-pathogen surveillance.

---

## Figures (the spine)

| Fig | File | Shows |
|---|---|---|
| 1 | `results/yolov8_bf_mar_to_nov/patient_roc.png` | Patient-level ROC, detector, bootstrap CI band, TPP points |
| 2 | `results/yolov8_bf_mar_to_nov/detection_vs_classification.png` | Forest plot: classifier within→cross-study collapse vs detector |
| 3 | `results/cross_study_mar_to_nov/classifier_saliency_nov.png` | Grad-CAM mechanism (classifier attends to background) |
| 4 | `results/chula_species_generalization/chula_species_generalization.png` | Cross-species generalization on Chula |

---

## Methods (brief)

- **Data.** Schistosomiasis brightfield microscopy, two field studies (mar2020 train/val,
  nov2021 test); patient-level clinical labels; egg point-annotations → fixed 150 px boxes;
  4032×3024 images tiled into 30×(640×640). Second dataset: Chula-ParasiteEgg-11 (COCO boxes,
  11 species), single-class collapse.
- **Detector.** YOLOv8s from COCO weights; single egg class; patient score = max detection
  confidence over all of a patient's tiles (de Leon Derby TPP aggregation).
- **Classifier baseline.** MobileNetV2 / EfficientNet-B0 (ImageNet), whole-image; patient
  score = max image probability.
- **Evaluation.** Patient-level AUC, sensitivity at 96.5% / 99.5% specificity; 2000-sample
  bootstrap 95% CIs. Same-direction, same-patient comparison.
- **Reproduce.** `scripts/train_yolov8_baseline.py`, `eval_yolov8.py`, `plot_patient_roc.py`,
  `cross_study_mar_to_nov_classifier.py`, `plot_detection_vs_classification.py`,
  `classifier_saliency_panel.py`, `plot_chula_species_generalization.py`.

---

## Honest limitations (state these; don't let a reviewer find them first)

- The high-level principle (classifiers exploit spurious shortcuts and fail out-of-distribution)
  is **known** in ML ("shortcut learning"). The contribution is the rigorous clinical
  demonstration + mechanism + cross-disease generalization, **not** the discovery of the
  phenomenon. Phrase accordingly; do not claim a new algorithm.
- Chula detection is **near-saturated / easy** (mAP50 0.99 in-distribution; ~1 large centered
  egg on clean background), so cross-species generalization partly reflects task ease + shared
  ovoid morphology. The honest "species matters" signal is the mAP50-95 drop (0.943→0.896).
  **One random fold** of held-out species (rotating folds = pending).
- Headline contrast is **one direction** (mar→nov). Prior runs show the classifier is also
  ≈chance nov→mar; the detector's *within-study* (mar→mar) AUC is not yet measured (needs a
  mar-only split + retrain).
- Mechanism (saliency) is **qualitative**; egg-overlap is not yet quantified.
- Darkfield contrast excluded (classifier ≈0.24, below chance — graveyard).
- **The detector result is not novel.** de Leon Derby et al. (2025) — whose *main* experiment is
  this exact train-March/test-Nov split (confirmed by co-author C. Delahunt) — already established
  cross-study BF detection (BF M&E sensitivity 76%; no patient-level AUC reported). Our detector
  **reproduces** that (M&E 73.8%, marginally lower). Do **not** claim it beats prior work. The
  contribution is the classifier-collapse **contrast** + mechanism + cross-species generalization
  + the (planned) method — not the detector.

---

## Status

This is the **empirical floor** — complete and defensible as-is. Planned next: convert the
*finding* into a *method* — a locality-constrained / attention-regularized classifier,
motivated directly by the mechanism above, that recovers cross-study robustness using only
cheap image-level labels (no boxes). To be de-risked cheaply before any large commitment;
if it fails, this document is the submission.
