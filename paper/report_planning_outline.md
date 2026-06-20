# STS Research Report — PLANNING OUTLINE (this is NOT the paper)

> **You must write all prose yourself, in your own words.** STS Rule #1: the report must be
> written **without generative AI**. This file is only an organizational scaffold of *your own
> results* + the format rules. Do not copy sentences from anywhere; write from your understanding.

## Format rules checklist (2027 Research Report Guidelines)
- [ ] ≤20 pages of content. Title page + abstract + bibliography are **excluded**; **appendices count**.
- [ ] Title page = page 1: your **name + project title** (required); mentor/lab optional; **no email/phone**.
- [ ] Abstract = page 2.
- [ ] Font **≥ Times New Roman 11pt**; **1.5 line spacing**; **1" margins**; **no columns**.
- [ ] Page numbers bottom-right, starting **after** the abstract.
- [ ] **Every graphic cited directly under/next to it** (APA), no reference list for graphics.
      Your own figures: "Figure created by the student researcher using Python/Matplotlib, 2026."
- [ ] References: build yourself, **no AI** (fake refs = disqualification); internal citations + bibliography at end.
- [ ] No links except in bibliography; no photos of people; "I" allowed if your field uses "we."
- [ ] PDF **≤4 MB**, file named **LASTNAME.FIRSTNAME.ZIPCODE**.
- [ ] Disclose **all support** (mentor + AI assistance) in the application.

## Suggested structure (you choose final order/format)

**Abstract (p2)** — one paragraph: the deployment problem (site shift + open-world), what you did, the headline numbers, the takeaway.

**Introduction** — points to cover: schistosomiasis burden + WHO NTD elimination; mobile microscopy diagnosis; the deployment problem (site shift, 10–25% degradation reported; open-world / novel species); the two framings (global classification vs. object detection); your research question; your contributions — *state explicitly that the detector reproduces de Leon Derby et al. (2025); your contribution is the comparison + mechanism + open-set + cheap-label method.*

**Methods** — data (two field studies, brightfield, 30×640px tiles, patient labels; Chula 11-species); classifier (MobileNetV2 / EfficientNet-B0, image labels, patient = max image prob); detector (YOLOv8s, box labels, patient = max detection conf); cross-study protocol (both directions, patient-disjoint); Chula held-out-species protocol (train 8, test 3 unseen, single seeded fold); MIL method (bag = FOV of 30 tiles, image-level label, max/attention pooling); evaluation (patient AUC + 2000× bootstrap CI; sens @96.5% & 99.5% spec; mAP50/50-95); Grad-CAM.

**Results** — the numbers to report (these are *your* data):
- R1 cross-study: classifier within-study 0.854 / 0.857 → cross-study **0.517 [0.432,0.600] / 0.504 [0.426,0.590]**; detector **0.914 [0.859,0.961]**; detector M&E sens 0.738 (reproduces de Leon Derby ≈76%); CIs non-overlapping. → **Figure 1**
- R2 mechanism: classifier outputs p≈0.04–0.16 on egg-present nov images; Grad-CAM on background not egg. → **Figure 2**
- R3 open-set (Chula): seen mAP50 **0.992** / mAP50-95 0.943; unseen 3 species **0.993** / 0.896. → **Figure 3**
- R4 locality recovery: whole-image **0.517** → local-MIL **0.711 (A→B) / 0.826 (B→A)**; max≈attention (0.711 vs 0.714). → **Figure 4**

**Discussion** — localize-the-invariant-target principle; open-world surveillance implication; annotation trade-off + cheap recovery; relation to prior work (reproduce de Leon Derby; phenomenon = known "shortcut learning").

**Limitations** — known principle/reproduction; Chula easy + single fold; small data (wide CIs, esp. TI&S); qualitative mechanism; darkfield excluded; MIL below detector and validated at classifier level only.

**Conclusion** — one paragraph restating the principle and the deployment/surveillance implication.

## Figures (cite each directly beneath it as student-created)
- Fig 1 `results/yolov8_bf_mar_to_nov/detection_vs_classification.png`
- Fig 2 `results/cross_study_mar_to_nov/classifier_saliency_nov.png`
- Fig 3 `results/chula_species_generalization/chula_species_generalization.png`
- Fig 4 `results/mil_locality_pulsecheck/method_pulsecheck.png`
- Supp `results/yolov8_bf_mar_to_nov/patient_roc.png`

## References to track down + cite yourself (APA, verify each — no AI)
de Leon Derby et al. 2025 (PLOS NTDs, multi-contrast schistosomiasis); Geirhos et al. 2020
(shortcut learning, Nature Machine Intelligence); ICIP 2022 Chula-ParasiteEgg-11; Jocher et al.
2023 (Ultralytics YOLOv8); Sandler et al. 2018 (MobileNetV2); Tan & Le 2019 (EfficientNet);
WHO schistosomiasis / NTD roadmap; a domain-shift-in-medical-imaging review.
