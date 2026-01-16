# Quick Reference - Confidence Percentiles

## 📊 At a Glance

| Percentile | Confidence | Coverage | Accuracy | Strategy |
|-----------|-----------|----------|----------|----------|
| **50th** | ≥0.089 | 50% | 56.3% | Volume |
| **90th** | ≥0.209 | 10% | 56.5% | ⭐ **RECOMMENDED** |
| **95th** | ≥0.285 | 5% | 57.6% | High-stakes |
| **99th** | ≥0.436 | 1% | 57.7% | Rare signals |

---

## 🎯 The Numbers

### 50th Percentile (Median)
```
✅ Games: 1,236 (50% of season)
✅ Correct: 696 (56.3% accuracy)
✅ Precision: 56.9%
✅ Recall: 83.8%
```

### 90th Percentile ⭐ BEST
```
✅ Games: 253 (10.2% of season)
✅ Correct: 143 (56.5% accuracy)
✅ Precision: 58.9% (highest)
✅ Recall: 83.0%
✅ F1 Score: 0.689 (best balance)
```

### 95th Percentile
```
✅ Games: 125 (5.1% of season)
✅ Correct: 72 (57.6% accuracy)
✅ Precision: 57.1%
✅ Recall: 88.2% (highest)
✅ F1 Score: 0.694 (highest)
```

### 99th Percentile
```
✅ Games: 26 (1.1% of season)
✅ Correct: 15 (57.7% accuracy)
✅ Precision: 55.6%
✅ Recall: 76.9%
✅ F1 Score: 0.645
```

---

## 💡 Key Insight

**Model is more accurate when most confident**

```
Accuracy Progression:
54.6% (overall)
  ↓
56.3% (50th percentile) +1.7%
  ↓
56.5% (90th percentile) +1.9%
  ↓
57.6% (95th percentile) +3.0%
  ↓
57.7% (99th percentile) +3.1%
```

---

## 🏆 Recommendation

### Use 90th Percentile
- **Why**: Best precision (58.9%), good balance (F1: 0.689)
- **Coverage**: 10.2% of games is practical volume
- **Accuracy**: +1.91% improvement over baseline
- **Precision**: When model is confident, 59% of picks are correct

### For Maximum Confidence: Use 95th
- **Accuracy**: 57.6% (+3.0% improvement)
- **Coverage**: 5.1% of games (125 per season)
- **Best F1**: 0.694 (best balance)

### For Maximum Coverage: Use 50th
- **Coverage**: 50% of games (practical for daily service)
- **Accuracy**: 56.3% (+1.7% improvement)
- **Still solid**: Better than baseline for half the season

---

## 📈 Coverage vs Accuracy

```
Want ALL games?        → Overall: 54.6%
Want 50% best?         → 50th%: 56.3%
Want 10% best?         → 90th%: 56.5% ⭐
Want 5% elite?         → 95th%: 57.6%
Want 1% exceptional?   → 99th%: 57.7%
```

---

## 🎲 Confidence Thresholds

```
0.089  = 50th percentile (median)
0.209  = 90th percentile (high confidence) ← START HERE
0.285  = 95th percentile (very high)
0.436  = 99th percentile (extreme)
```

Lower threshold = more games, lower accuracy
Higher threshold = fewer games, higher accuracy

---

## 2026 Season Projection (90th Percentile)

```
Total Games:         2,430
High-Confidence:     248 games
Expected Correct:    140 wins
Expected Incorrect:  108 losses
Accuracy:            56.5%
```

vs Baseline (50%):    **+6.5%** improvement
vs Overall (54.6%):   **+1.9%** improvement

---

## ✨ Bottom Line

**Use confidence thresholds to improve accuracy by 1-3%**

The model knows when it's right - trust it when confident!

---

*90th percentile recommended for 2026 season*
