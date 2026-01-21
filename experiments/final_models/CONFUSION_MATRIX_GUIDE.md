# How to Report Confusion Matrices in Your Dissertation

## 1. Generate the Figure

Run the script:
```bash
cd experiments/final_models
python generate_confusion_matrices.py
```

This will create:
- `results/figures/confusion_matrices_all_models.png` (for viewing)
- `results/figures/confusion_matrices_all_models.pdf` (for LaTeX)

---

## 2. Where to Place It in Your Dissertation

### **Option A: Main Results Section** (RECOMMENDED)

Place immediately after Table 1 (aggregate performance metrics):

```markdown
## 4. Results

### 4.1 Overall Model Performance

**Table 1: Model Performance Summary**
[Your existing table]

**Figure 1: Confusion Matrices for All Models**
[Insert confusion matrix figure here]
```

---

### **Option B: Supplementary Material**

If your Results section is already dense, place in appendix:

```markdown
## Appendix A: Detailed Model Performance

**Figure A1: Confusion Matrices**
[Insert confusion matrix figure here]
```

---

## 3. Figure Caption (What to Write)

### **Full Caption:**

```
Figure X: Confusion matrices for all models using G-mean threshold optimization 
(B1 strategy). Each cell shows the proportion of predictions (0-1 scale) with 
raw counts (n=X) below. Matrices are normalized by true class (rows sum to 1). 
All models evaluated using LOSO cross-validation on 1,733 windows from 21 subjects. 
Key observations: (1) LR and Stacked achieve highest true positive rates (0.72, 0.74) 
but moderate false positive rates (0.39, 0.30); (2) TCN shows highest true negative 
rate (0.85) but lowest true positive rate (0.43); (3) RF balances true positive 
(0.50) and true negative (0.81) rates.
```

---

### **Shorter Caption (if space limited):**

```
Figure X: Confusion matrices for all models (G-mean threshold, LOSO validation). 
Cells show proportions (normalized by true class) with raw counts (n=X). 
LR and Stacked achieve highest recall (TP rate: 0.72, 0.74), while TCN minimizes 
false alarms (FP rate: 0.15).
```

---

## 4. How to Reference in Text

### **In Results Section:**

**Example 1: Introducing the figure**
> "Figure 1 presents confusion matrices for all five models. Logistic Regression 
and Stacked Generalization achieved the highest true positive rates (0.72 and 0.74, 
respectively), detecting approximately 3 out of 4 stress events. However, this came 
at the cost of elevated false positive rates (0.39 and 0.30), meaning approximately 
30-40% of non-stress windows were misclassified as stress."

**Example 2: Comparing models**
> "The confusion matrices (Figure 1) reveal distinct operating characteristics across 
models. TCN exhibited the most conservative behavior, with a true negative rate of 
0.85 (correctly identifying 85% of non-stress periods) but a true positive rate of 
only 0.43 (detecting 43% of stress events). In contrast, Logistic Regression prioritized 
recall, achieving a true positive rate of 0.72 at the expense of a higher false positive 
rate of 0.39."

**Example 3: Discussing trade-offs**
> "As shown in Figure 1, no model simultaneously maximized both true positive and true 
negative rates, reflecting the fundamental recall-FAR trade-off in imbalanced classification. 
Random Forest struck the best balance, with moderate rates in both quadrants (TP=0.50, TN=0.81), 
while LR and Stacked prioritized recall (TP=0.72-0.74) over specificity (TN=0.61-0.70)."

---

### **In Discussion Section:**

**Example: Clinical implications**
> "The confusion matrix analysis (Figure 1) provides insight into deployment trade-offs. 
For clinical mental health applications, where missing a stress event (false negative) 
is costly, Stacked Generalization's 74% true positive rate may be preferred despite a 
30% false positive rate. Conversely, for consumer wellness applications sensitive to user 
annoyance, TCN's 15% false positive rate may justify its lower 43% true positive rate."

---

## 5. What to Highlight in Each Quadrant

Use this as a guide when discussing the matrices:

### **Top-Left (True Negative, TN)**
- **What it means**: % of non-stress windows correctly identified
- **High TN = Good**: Low false alarm rate
- **Best model**: TCN (0.85), RF (0.81)
- **Clinical interpretation**: "System doesn't annoy users with frequent false alarms"

### **Top-Right (False Positive, FP)**
- **What it means**: % of non-stress windows misclassified as stress
- **Low FP = Good**: Fewer false alarms
- **Worst model**: LR (0.39), SVM (0.35)
- **Clinical interpretation**: "Too many false alarms may reduce user trust"

### **Bottom-Left (False Negative, FN)**
- **What it means**: % of stress windows missed
- **Low FN = Good**: Fewer missed stress events
- **Best model**: Stacked (0.26), LR (0.28)
- **Clinical interpretation**: "Missing stress events prevents early intervention"

### **Bottom-Right (True Positive, TP)**
- **What it means**: % of stress windows correctly detected
- **High TP = Good**: High recall
- **Best model**: Stacked (0.74), LR (0.72)
- **Clinical interpretation**: "System provides timely warnings before stress onset"

---

## 6. Common Mistakes to Avoid

### ❌ **Don't Say:**
> "LR achieved 0.72 accuracy on stress detection"

**Why wrong:** 0.72 is the recall (TP rate), not accuracy. Accuracy would include all four quadrants.

### ✅ **Do Say:**
> "LR achieved 0.72 recall (true positive rate), correctly identifying 72% of stress events"

---

### ❌ **Don't Say:**
> "TCN has the best confusion matrix because all cells are high"

**Why wrong:** TCN's TP rate is only 0.43 (lowest). High TN (0.85) doesn't make it "best" overall.

### ✅ **Do Say:**
> "TCN optimizes for specificity (TN rate: 0.85), achieving the lowest false alarm rate 
at the cost of reduced recall (TP rate: 0.43)"

---

### ❌ **Don't Say:**
> "The confusion matrix shows accuracy"

**Why wrong:** Confusion matrix shows **4 separate metrics** (TN, FP, FN, TP). Accuracy is a derived metric: (TN + TP) / Total.

### ✅ **Do Say:**
> "The confusion matrix shows the distribution of true/false positives/negatives, 
revealing the recall-specificity trade-off"

---

## 7. How to Interpret Normalized vs. Raw Counts

Your figure shows **BOTH** - this is good!

### **Normalized (0.72)**:
- **Meaning**: 72% of actual stress windows were detected
- **Use**: Compare model behavior (which prioritizes recall vs. specificity?)
- **Benefit**: Accounts for class imbalance (12% stress, 88% non-stress)

### **Raw Counts (n=146)**:
- **Meaning**: 146 stress windows detected out of 208 total
- **Use**: Understand absolute numbers (how many predictions?)
- **Benefit**: Shows actual impact (e.g., "detects 146 stress events")

**Both are necessary:**
- Proportions → understanding model behavior
- Raw counts → understanding real-world impact

---

## 8. LaTeX Code (If Needed)

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/confusion_matrices_all_models.pdf}
  \caption{Confusion matrices for all models using G-mean threshold optimization 
  (B1 strategy). Each cell shows the proportion of predictions (0-1 scale) with 
  raw counts (n=X) below. Matrices are normalized by true class (rows sum to 1). 
  All models evaluated using LOSO cross-validation on 1,733 windows from 21 subjects.}
  \label{fig:confusion_matrices}
\end{figure}

As shown in Figure~\ref{fig:confusion_matrices}, Logistic Regression and 
Stacked Generalization achieved the highest recall (TP rates: 0.72 and 0.74).
```

---

## 9. Summary Checklist

Before including your confusion matrix figure:

- [ ] Remove Logical OR and Cascade (not used in your study)
- [ ] Include all 5 models: LR, RF, SVM, TCN, Stacked
- [ ] Show normalized proportions (0-1 scale)
- [ ] Show raw counts (n=X)
- [ ] Use consistent color scheme across models
- [ ] Include clear axis labels
- [ ] Add descriptive caption explaining normalization
- [ ] Reference in text with interpretation
- [ ] Highlight key trade-offs (recall vs. FAR)

---

## 10. Quick Interpretation Guide

**For your dissertation, focus on these key points:**

1. **Stacked has best TP rate (0.74)**: "Detects 3 out of 4 stress events"
2. **TCN has best TN rate (0.85)**: "Lowest false alarm rate (15%)"
3. **LR nearly matches Stacked (TP=0.72)**: "Simple model competitive with fusion"
4. **RF balances both (TP=0.50, TN=0.81)**: "Best precision-recall balance"
5. **No model perfect**: "All exhibit recall-FAR trade-off inherent to imbalanced data"

This framework helps you discuss results clearly and avoid common pitfalls!
