#!/usr/bin/env python3
"""Build a single clean PDF (summary + four figures inline) for sharing/review.

Output: summary_localize_dont_classify.pdf  (in repo root)
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "summary_localize_dont_classify.pdf"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=15, leading=18, spaceAfter=2)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=8.5, textColor=colors.grey, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#1A3C6E"))
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=9.5, leading=13, spaceAfter=6)
CAP = ParagraphStyle("CAP", parent=ss["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#444444"), spaceBefore=2, spaceAfter=12)
BUL = ParagraphStyle("BUL", parent=ss["Normal"], fontSize=9, leading=12, leftIndent=10, bulletIndent=0, spaceAfter=4)


def figure(path: Path, caption: str, max_w=6.6 * inch, max_h=3.7 * inch):
    im = PILImage.open(path); w, h = im.size; ar = h / w
    iw, ih = max_w, max_w * ar
    if ih > max_h:
        ih, iw = max_h, max_h / ar
    return KeepTogether([Image(str(path), width=iw, height=ih),
                         Paragraph(caption, CAP)])


def table(data, col_widths):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A3C6E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2F7")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


story = []
story.append(Paragraph("Localize, don’t classify: detection survives the study shift that collapses classification", H1))
story.append(Paragraph("Schistosomiasis (mar2020→nov2021, brightfield) + Chula-ParasiteEgg-11 (11 species) · summary for review · 2026-06-14", SUB))

story.append(Paragraph("Summary", H2))
story.append(Paragraph(
    "Automated microscopy diagnosis is usually framed as <b>image classification</b> (is this sample positive?). "
    "On schistosomiasis brightfield microscopy across two field studies (train mar2020, zero-shot test nov2021), an "
    "ImageNet-pretrained classifier learns the source study well (within-study AUC ≈ 0.85) but <b>collapses to chance "
    "on the unseen study (AUC ≈ 0.52, 95% CI through 0.50)</b>. Re-framing the task as <b>object detection</b> (localize "
    "the egg, aggregate per patient) is robust to the same shift: a YOLOv8 detector on the <i>same</i> patients holds at "
    "AUC 0.914 [0.859, 0.961] — <b>reproducing the established cross-study result of de Leon Derby et al. (2025)</b> "
    "(their BF M&amp;E sensitivity 76%; ours 73.8%), not beating it. Saliency shows the mechanism: the classifier attends "
    "to global background / illumination, not the egg. On a second dataset (Chula, 11 species), a class-agnostic egg "
    "detector trained on 8 species generalizes to 3 unseen species (mAP50 0.993 vs 0.992). "
    "<b>The contribution is the contrast + mechanism + cross-species generalization — localizing the invariant target is "
    "robust to shift (site shift and novel-species shift) that breaks global classification.</b>", BODY))

story.append(Paragraph("Result 1 — classification collapses cross-study; detection does not (same direction, same patients)", H2))
story.append(table(
    [["Model", "within-study (mar val)", "cross-study (nov, zero-shot)"],
     ["MobileNetV2", "0.854", "0.517 [0.432, 0.600]"],
     ["EfficientNet-B0", "0.857", "0.504 [0.426, 0.590]"],
     ["YOLOv8s detector", "—", "0.914 [0.859, 0.961]"]],
    [2.0 * inch, 2.0 * inch, 2.4 * inch]))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "380 nov2021 patients (61 pos / 319 neg). Detector M&amp;E sensitivity 0.738 @ 96.5% spec (de Leon Derby BF: 0.76); "
    "TI&amp;S 0.557 @ 99.5% spec. Classifier and detector CIs do not overlap. 2000-sample bootstrap CIs.", BODY))

story.append(Paragraph("Result 2 — cross-species generalization (Chula-ParasiteEgg-11)", H2))
story.append(table(
    [["", "in-distribution (8 seen)", "out-of-distribution (3 unseen)"],
     ["mAP@50", "0.992", "0.993"],
     ["mAP@[.5:.95]", "0.943", "0.896"]],
    [2.0 * inch, 2.0 * inch, 2.4 * inch]))

story.append(Paragraph("Figures", H2))
story.append(figure(REPO / "results/yolov8_bf_mar_to_nov/detection_vs_classification.png",
                    "Fig 1. Same patients, same shift: the classifier learns mar2020 (0.85) but collapses to chance on "
                    "nov2021 (0.52); the detector holds (0.914)."))
story.append(figure(REPO / "results/yolov8_bf_mar_to_nov/patient_roc.png",
                    "Fig 2. Patient-level ROC of the detector (mar→nov, bootstrap CI band, M&E/TI&S operating points)."))
story.append(figure(REPO / "results/cross_study_mar_to_nov/classifier_saliency_nov.png",
                    "Fig 3. Mechanism — Grad-CAM of the cross-study classifier on egg-present nov2021 images: attention "
                    "on background/illumination, not the egg."))
story.append(figure(REPO / "results/chula_species_generalization/chula_species_generalization.png",
                    "Fig 4. Second disease: a single-class detector trained on 8 species localizes 3 unseen species "
                    "(mAP50 0.993 vs 0.992 in-distribution)."))

story.append(Paragraph("Honest limitations", H2))
for b in [
    "The high-level principle (classifiers exploit shortcuts and fail out-of-distribution) is known in ML; the "
    "contribution is the rigorous clinical demonstration + mechanism + cross-disease generalization, not a new phenomenon.",
    "The detector result reproduces de Leon Derby et al. (2025) (same March→Nov split); it is not claimed to beat prior work.",
    "Chula detection is near-saturated/easy (mAP50 0.99 in-distribution); the honest signal is the mAP50-95 drop "
    "(0.943→0.896). One random fold of held-out species.",
    "Headline contrast is one direction (mar→nov); detector within-study (mar→mar) not yet measured. Mechanism is qualitative.",
]:
    story.append(Paragraph("• " + b, BUL))

doc = SimpleDocTemplate(str(OUT), pagesize=letter, topMargin=0.6 * inch,
                        bottomMargin=0.6 * inch, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        title="Localize, don't classify — summary")
doc.build(story)
print("Saved:", OUT)
