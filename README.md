# Do VLMs Actually Use Text Context — or Just the Image?

A controlled experiment evaluating whether vision-language models genuinely use text context for action recognition, or rely purely on visual features.

Tested on **three datasets** of increasing difficulty using **Qwen2-VL 7B**.

---

## The Question

When you pass a label or description alongside an image to a VLM, does the model actually use it — or does it just look at the image and ignore the text?

This matters practically: if you are building a multimodal pipeline and passing action labels, class names, or descriptions as context, are they doing anything?

---

## Experimental Design

Three conditions, same image each time, only the prompt changes:

| Condition | Input | Purpose |
|-----------|-------|---------|
| **A — Zero-shot** | Frame only | Baseline: raw visual understanding |
| **B — Oracle Grounded** | Frame + correct label | Upper bound: does accurate context help? |
| **C — Noise Injection** | Frame + wrong label | Control: does the model genuinely use context? |

Condition C is the key design decision. If wrong context degrades accuracy below zero-shot, the model is genuinely reading and using the text — not ignoring it.

**Model:** Qwen2-VL 7B
**Frame selection:** Single middle frame extracted from each video
**Significance testing:** McNemar test on paired binary outcomes

---

## Datasets

| Dataset | Classes | Format | Difficulty | Samples |
|---------|---------|--------|------------|---------|
| UCF-101 | 101 | .avi | Low — scene-based actions | 303 |
| Kinetics-400 Mini | 200 | .mp4 | Medium — diverse YouTube clips | 400 |
| Something-Something v2 | 174 | .webm | High — requires temporal reasoning | 348 |

---

## Results

| Dataset | A Zero-shot | B Oracle | C Noise | B-A delta | C-A delta |
|---------|------------|---------|---------|-----------|-----------|
| UCF-101 | 0.749 | 1.000 | 0.640 | **+0.251** | -0.109 |
| Kinetics-400 Mini | 0.642 | 0.993 | 0.487 | **+0.350** | -0.155 |
| Something-Something v2 | 0.075 | 0.804 | 0.037 | **+0.729** | -0.037 |

All differences statistically significant (McNemar test, p < 0.0001 across all datasets and condition pairs).

### Key Findings

**Finding 1: VLMs genuinely use text context.**
Oracle grounding improves accuracy significantly on all three datasets. The model is not ignoring the prompt.

**Finding 2: Wrong context actively hurts performance.**
Noise injection degrades accuracy below zero-shot on all three datasets. The model reads the label and is misled by incorrect ones.

**Finding 3: The grounding effect scales with dataset difficulty.**
As zero-shot accuracy drops (harder dataset), the B-A improvement grows larger. The model relies on context more when visual information is ambiguous.

| Dataset | Zero-shot | Grounding improvement |
|---------|-----------|----------------------|
| UCF-101 | 74.9% | +25.1% |
| Kinetics-400 Mini | 64.2% | +35.0% |
| Something-Something v2 | 7.5% | +72.9% |

**Note on Something-Something v2:** This dataset is explicitly designed to require temporal reasoning. Actions like "moving something up" vs "moving something down" are visually identical from a single frame. The 7.5% zero-shot reflects the single-frame limitation, not model capability. Oracle accuracy of 80.4% shows the model understands these actions correctly when given context. A multi-frame extension is planned.

---

## Practical Implication

If you build multimodal systems, the labels and descriptions you pass as context are not decorative. The model reads them — and wrong or noisy context actively degrades performance. This has direct implications for prompt design in production VLM pipelines.

---

## Notebooks

| Notebook | Dataset | Kaggle |
|----------|---------|--------|
| `notebooks/ucf101_eval.ipynb` | UCF-101 | [Open in Kaggle](#) |
| `notebooks/kinetics400_eval.ipynb` | Kinetics-400 Mini | [Open in Kaggle](#) |
| `notebooks/multi_dataset_eval.ipynb` | SSv2 + config switcher | [Open in Kaggle](#) |

---

## Repository Structure

```
vlm-action-recognition/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── ucf101_eval.ipynb
│   ├── kinetics400_eval.ipynb
│   └── multi_dataset_eval.ipynb
└── results/
    ├── ucf101_results.csv
    ├── kinetics400_results.csv
    ├── ssv2_results.csv
    └── figures/
        ├── ucf101_results.png
        ├── kinetics400_results.png
        ├── ssv2_results.png
        └── cross_dataset_comparison.png
```

---

## Reproduce

```bash
pip install -r requirements.txt
```

Open any notebook in `notebooks/` and follow the dataset setup instructions at the top. All experiments run on a single T4 GPU (Kaggle free tier).

To switch datasets in `multi_dataset_eval.ipynb`, change one line:

```python
ACTIVE_DATASET = 'ssv2'        # or 'ucf101' or 'kinetics400'
```

---

## Limitations

- **Single frame per video** — middle frame only. Multi-frame evaluation is a planned extension, particularly for SSv2.
- **Substring match scoring** — hit scoring uses case-insensitive substring match, not semantic similarity. May underestimate zero-shot accuracy when the model paraphrases correctly.
- **2 samples per class** — per-class results should be interpreted cautiously at this sample size.
- **Closed-set evaluation** — all class names are provided in the prompt for all conditions including zero-shot. This is not open-vocabulary evaluation.

---

## Planned Extensions

- [ ] Multi-frame condition for SSv2 (4 evenly spaced frames)
- [ ] Semantic similarity scoring using sentence-transformers
- [ ] Second model comparison (LLaVA or InternVL)
- [ ] Pixtral 12B substitution when VRAM available

---

## Author

**Muneeba Nasir** — ML Engineer at CNRS I3S, Sophia Antipolis
[LinkedIn](#) · [Kaggle](#)
