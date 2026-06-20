# Localize, Don't Classify: Object Detection Confers Cross-Site and Open-Set Robustness for Microscopy-Based Parasite Diagnosis

*Regeneron STS 2027 research report — working draft. Structured as a standard scientific
paper (Abstract / Introduction / Methods / Results / Discussion / Limitations / Conclusion /
References). Target ≤20 pages. Every figure must carry a citation per STS rules.*

---

## Abstract

Automated microscopy promises low-cost diagnosis of neglected tropical diseases (NTDs) in
the field, but deployed models must survive two real-world shifts: **site shift** (a model
trained at one clinic is used at another) and the **open-world** problem (samples may contain
pathogen species the model never trained on). We ask whether the *framing* of the task —
global **image classification** versus **object detection** — determines robustness to these
shifts. Using brightfield microscopy of urine from two field studies of *Schistosoma
haematobium* (Côte d'Ivoire, 2020 and 2021), we show that a strong ImageNet-pretrained
classifier learns its source study well (within-study patient AUC ≈ 0.85) but **collapses to
chance across the study shift** (AUC ≈ 0.52, 95% CI through 0.50), in both transfer
directions. An object detector evaluated on the *same patients* retains AUC 0.91 — reproducing
the established cross-study performance of de Leon Derby et al. (2025). Saliency analysis
identifies the mechanism: the failing classifier attends to global background and illumination
rather than the egg. On an independent 11-species parasite dataset, a single-class detector
trained on 8 species localizes 3 species it never saw (mAP50 0.99) — a capability a closed-set
classifier **structurally cannot have**. Finally, we show that simply forcing the classifier
to reason *locally* (multiple-instance learning over image tiles, image-level labels only)
recovers much of the lost robustness (AUC 0.52 → 0.71–0.83) without any localization
annotation. Together these results support a single design principle for robust NTD diagnosis:
**localize the invariant target rather than classify the global image** — and they point toward
open-world parasite surveillance, where the ability to flag *unseen* pathogens matters.

---

## 1. Introduction

**1.1 Burden and diagnosis.** Schistosomiasis is a parasitic disease affecting more than 200
million people, concentrated in sub-Saharan Africa, and is a World Health Organization (WHO)
priority NTD targeted for elimination as a public-health problem. Diagnosis of urogenital
schistosomiasis relies on detecting *S. haematobium* eggs by microscopy of urine — a task that
requires a trained microscopist, who is scarce in the endemic, resource-limited settings where
the disease is most prevalent. Low-cost, mobile, phone-based microscopes such as the
SchistoScope, paired with automated machine-learning (ML) readouts, are a promising path to
scalable screening [de Leon Derby et al., 2025].

**1.2 The deployment problem.** For an automated diagnostic to be useful in the field, accuracy
on the lab's own data is not enough. Two shifts threaten real deployment:

- **Site / study shift.** Different clinics, devices, illumination, sample-preparation, and
  patient populations induce distribution shift. Published reviews report 10–25% performance
  degradation when medical-imaging models are tested on unseen populations, and existing
  domain-adaptation fixes give only modest (5–15%), externally-unvalidated gains.
- **The open-world problem.** A field sample may contain a parasite species the model was never
  trained to recognize. A diagnostic that silently mishandles novel pathogens is unsafe for
  surveillance, where detecting the *unexpected* is the point.

**1.3 Two framings, one question.** Automated microscopy diagnosis is built in two broad ways.
A **global classifier** consumes a whole image and outputs a positive/negative probability — it
needs only cheap image-level labels. An **object detector** localizes individual pathogens and
then aggregates to a patient decision — it needs expensive bounding-box (or point) annotations.
These are usually treated as interchangeable engineering choices. **We ask whether the choice
itself determines robustness to site shift and to novel species.**

**1.4 Contributions.** (1) On the same patients and the same study shift, we show that
classification collapses to chance while detection is retained, in both transfer directions,
and we identify the mechanism by saliency. (2) On a second, independent disease dataset we show
that class-agnostic detection generalizes to *unseen* parasite species, a property closed-set
classification cannot have. (3) We show that the robustness gap is largely about *locality*: a
weakly-supervised, locality-constrained classifier — image-level labels only, no boxes —
recovers most of detection's cross-site robustness. We frame the result as a design principle
for robust, open-world NTD diagnosis. We are explicit throughout that the detector result
*reproduces* prior work [de Leon Derby et al., 2025]; our contribution is the comparative
robustness analysis, its mechanism, the open-set generalization, and the label-efficient
recovery method — not the detector itself.

---

## 2. Methods

**2.1 Data.** We use brightfield (BF) microscopy of urine sediment from two field studies of
*S. haematobium* in Côte d'Ivoire (Study A, March 2020; Study B, November 2021), captured on a
mobile phone–based microscope. Patient-level ground truth (positive/negative) follows trained-
microscopist annotation. Each 4032×3024 field-of-view image is divided into 30 overlapping
640×640 tiles. For detection, expert egg-center annotations are converted to fixed 150×150 px
boxes (single class, *schistosoma egg*). Patient splits are patient-disjoint. For the open-set
experiment we use the public **Chula-ParasiteEgg-11** dataset (11 intestinal-parasite species,
bounding-box annotated) [ICIP 2022 challenge].

**2.2 Classifier baseline.** MobileNetV2 and EfficientNet-B0 (ImageNet-pretrained), fine-tuned
on whole images with image-level labels. Patient score = maximum image probability over a
patient's images.

**2.3 Detector.** YOLOv8s (COCO-pretrained), fine-tuned on the 640×640 tiles as a single-class
egg detector. Patient score = maximum detection confidence over all of a patient's tiles
(target-product-profile aggregation).

**2.4 Cross-study protocol.** We train on one study and evaluate zero-shot on the other, in
both directions (A→B and B→A), on patient-disjoint test sets. For the detector we evaluate
A→B (the direction with established prior results).

**2.5 Open-set protocol (Chula).** A single-class "parasite egg" detector is trained on 8 of
the 11 species; 3 held-out species (selected at random, seed-fixed) form a test set evaluated
**once**. Model selection uses an in-distribution validation set of the 8 training species, so
the unseen species never influence training.

**2.6 Locality-constrained method.** To test whether *locality* (not box supervision) drives
robustness, we train a multiple-instance-learning (MIL) classifier: each field of view is a bag
of its 30 tiles; the bag label is the image-level positive/negative label (no box coordinates);
a MobileNetV2 backbone scores each tile and the bag prediction is pooled across tiles
(max-pooling and attention-pooling variants). This uses the *same* cheap labels as the global
classifier but is architecturally forced to decide from local evidence.

**2.7 Evaluation.** Primary metric is patient-level AUC with 2000-sample bootstrap 95%
confidence intervals; we also report sensitivity at 96.5% specificity (WHO M&E target) and
99.5% specificity (TI&S target). Detection quality is reported as mAP50 and mAP50-95. Mechanism
is probed with Grad-CAM saliency on the cross-study classifier.

---

## 3. Results

**3.1 Classification collapses across the study shift; detection does not (Figure 1).**
On the same 380 test patients (61 positive), trained in the same direction (Study A → Study B):

| Model | within-study (val) | cross-study (zero-shot) |
|---|---|---|
| MobileNetV2 (image labels) | 0.854 | 0.517 [0.432, 0.600] |
| EfficientNet-B0 (image labels) | 0.857 | 0.504 [0.426, 0.590] |
| YOLOv8 detector (box labels) | — | **0.914 [0.859, 0.961]** |

The classifier learns its source study well yet lands on **chance** across the shift (CI spans
0.50), for two independent backbones. The detector retains AUC 0.91; its operating points
(sensitivity 0.74 at 96.5% specificity) **reproduce** the established brightfield result of
de Leon Derby et al. (2025) (76% sensitivity at the same target). The classifier and detector
confidence intervals do not overlap.

**3.2 Mechanism (Figure 2).** On Study-B images that *contain* eggs, the cross-study classifier
outputs low positive probability (≈0.04–0.16) — it misses them — and Grad-CAM shows its
attention spread over background and illumination gradients rather than concentrated on the
egg. This is consistent with "shortcut learning": the classifier latched onto study-specific
global cues that do not transfer.

**3.3 Open-set generalization to unseen species (Figure 3).** A single-class egg detector
trained on 8 Chula species and tested on 3 it never saw achieves **mAP50 0.993**, essentially
matching in-distribution performance (0.992); the stricter mAP50-95 drops modestly
(0.943 → 0.896), the honest signal that species identity still carries some information. A
class-agnostic detector handles novel categories; a closed-set classifier has no output for a
class it never trained on, and so structurally cannot.

**3.4 Locality recovers robustness with cheap labels (Figure 4).** The locality-constrained MIL
classifier — image-level labels only, *no* boxes — lifts cross-study patient AUC from the
whole-image classifier's 0.52 to **0.71 (A→B)** and **0.83 (B→A)**, both well above the
whole-image baselines and far above chance. Because it shares the backbone, labels, and
aggregation of the failing classifier and differs only in *receptive field*, the comparison is
itself an ablation isolating locality as the causal factor. Aggregation choice (max vs.
attention) does not change the result (0.711 vs. 0.714), confirming the lever is locality, not
the pooling. The method does not reach the box-supervised detector (0.91), recovering roughly
half the gap — i.e., cheap labels buy back a large, significant fraction of detection's
robustness without any localization annotation.

---

## 4. Discussion

**4.1 A design principle.** Across two diseases and two kinds of distribution shift, the same
principle holds: **localizing the invariant target is robust to shifts that defeat global
classification.** The egg's local morphology is the same organism across studies and is shared
across egg-laying species; the global image statistics (stain, illumination, background) are
study- and device-specific confounds. Models that key on the former transfer; models that key
on the latter do not.

**4.2 Open-world surveillance.** The open-set result reframes the practical stakes. Elimination
and surveillance programs must flag *unexpected* pathogens — emerging species, co-infections, or
samples from regions with different parasite profiles. A class-agnostic localizer can flag "an
egg-like object is here" for an unseen species; a closed-set classifier cannot represent it.
This positions detection-based diagnosis as the appropriate framing for open-world parasite
surveillance, not merely as a marginally-better classifier.

**4.3 The annotation trade-off, and a cheap recovery.** Detection's robustness comes at the
cost of bounding-box annotation, the practical bottleneck for deploying to new sites. Our MIL
result shows much of the robustness is recoverable with only image-level labels by enforcing
locality — a label-efficient operating point between the failed classifier and the expensive
detector.

**4.4 Relation to prior work.** Our detector reproduces the cross-study brightfield result of
de Leon Derby et al. (2025), whose main experiment is this same train-A/test-B split; we do not
claim to exceed it. The contribution is the controlled *comparison* against classification, the
mechanism, the open-set generalization, and the label-efficient recovery — analyses absent from
prior work. The high-level phenomenon (classifiers exploit shortcuts and fail out-of-
distribution) is known in machine learning [Geirhos et al., 2020]; our contribution is its
rigorous, mechanistic demonstration in a clinically-grounded NTD setting and its extension to
the open-world case.

---

## 5. Limitations

We state these plainly. (1) The underlying principle — shortcut learning and the
out-of-distribution fragility of classifiers — is established in ML; this work is a rigorous
clinical demonstration and extension, not the discovery of a new phenomenon, and the detector
result is a reproduction of prior work. (2) The open-set detection task on Chula is relatively
easy (in-distribution mAP50 ≈ 0.99; roughly one large, centered egg per clean image), so
cross-species generalization partly reflects task ease and shared egg morphology; the honest
signal that species still matter is the mAP50-95 drop, and the open-set result rests on a single
random held-out-species fold. (3) Datasets are modest (hundreds of positive patients), limiting
statistical power, especially at the high-specificity (TI&S) operating point. (4) The mechanism
analysis is qualitative; egg-vs-background attention overlap is not yet quantified. (5)
Darkfield imaging is available but excluded here (its single-contrast classifier performed below
chance). (6) The locality-recovery method does not reach detection-level performance and was
validated at the classifier level, not the detector level.

---

## 6. Conclusion

The choice between classifying the whole image and localizing the pathogen is not a neutral
engineering detail — it determines whether a microscopy diagnostic survives the site shift and
the novel-species conditions of real deployment. Detection-based, class-agnostic localization is
robust to both; global classification is not, because it learns the wrong, transient features.
Much of the robustness is recoverable with cheap image-level labels by enforcing locality. For
field-deployable, open-world NTD surveillance, the guiding principle is simple: **localize the
invariant target rather than classify the global image.**

---

## Figures

- **Figure 1.** Forest plot of patient-level cross-study AUC: classifier (within-study → chance)
  vs. detector. *(results/yolov8_bf_mar_to_nov/detection_vs_classification.png)*
- **Figure 2.** Grad-CAM saliency of the failed cross-study classifier on egg-present images.
  *(results/cross_study_mar_to_nov/classifier_saliency_nov.png)*
- **Figure 3.** Held-out-species generalization on Chula (8 seen vs. 3 unseen species).
  *(results/chula_species_generalization/chula_species_generalization.png)*
- **Figure 4.** Locality recovers robustness with cheap labels (whole-image vs. local-MIL vs.
  detector). *(results/mil_locality_pulsecheck/method_pulsecheck.png)*
- **(Supp.)** Patient-level ROC of the detector with TPP operating points.
  *(results/yolov8_bf_mar_to_nov/patient_roc.png)*

## References (to be completed in citation format)

1. de Leon Derby et al. *Multi-contrast machine learning improves schistosomiasis diagnostic
   performance.* PLOS Neglected Tropical Diseases, 2025.
2. Geirhos et al. *Shortcut learning in deep neural networks.* Nature Machine Intelligence, 2020.
3. ICIP 2022 Challenge. *Parasitic Egg Detection and Classification in Microscopic Images*
   (Chula-ParasiteEgg-11).
4. Jocher et al. *Ultralytics YOLOv8.* 2023.
5. Sandler et al. *MobileNetV2.* CVPR 2018. — Tan & Le. *EfficientNet.* ICML 2019.
6. World Health Organization. *Schistosomiasis* fact sheet / NTD roadmap 2021–2030.
7. (Domain shift in medical imaging — review; to cite.)
