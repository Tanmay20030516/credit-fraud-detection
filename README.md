# Credit Card Fraud Detection: Class Imbalance Analysis

## Overview

This project presents a comprehensive analysis of credit card fraud detection, systematically comparing multiple approaches to handle severe class imbalance. The study evaluates **11 different models** ranging from baseline approaches to sophisticated combinations of resampling techniques, clustering methods, and class weighting strategies.

**Key Finding:** Advanced resampling techniques combined with strategic class weight tuning can achieve performance comparable to complex ensemble methods, with Logistic Regression + CBU + GMM + Class Weights reaching an F1-score of **0.808** - demonstrating that sophisticated data preprocessing can match the performance of more complex algorithms.

## Dataset

**Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Class Imbalance:** Only 0.17% fraudulent transactions (~400 fraud cases)
- **Challenge:** Extreme imbalance requiring sophisticated handling techniques

## Methodology & Model Comparison

### Models Evaluated

| Model | Precision | Recall | F1-Score | Key Characteristics |
|-------|-----------|--------|----------|-------------------|
| **Baseline Logistic Regression** | 0.817 | 0.662 | 0.731 | Strong precision, moderate recall |
| **LR + Class Weights** | 0.740 | 0.730 | 0.735 | Simple improvement over baseline |
| **LR + SMOTE** | 0.063 | 0.892 | 0.118 | High recall, precision collapse |
| **LR + SMOTE + Class Weights** | 0.713 | 0.770 | 0.740 | Balanced improvement |
| **LR + CBO** | 0.090 | 0.878 | 0.164 | Similar to SMOTE issues |
| **LR + CBO + Class Weights** | 0.820 | 0.676 | 0.741 | Better balanced performance |
| **LR + CBU** | 0.035 | 0.851 | 0.068 | Worst precision performance |
| **LR + CBU + Class Weights** | 0.680 | 0.689 | 0.685 | Moderate improvement |
| **LR + CBU + GMM** | 0.093 | 0.851 | 0.168 | Poor precision despite GMM |
| **🏆 LR + CBU + GMM + Class Weights** | **0.819** | **0.797** | **0.808** | **Optimal preprocessing approach** |
| **Random Forest + CBU + GMM** | **0.932** | **0.743** | **0.827** | Complex ensemble method |

### Resampling Techniques Explored

1. **SMOTE (Synthetic Minority Oversampling Technique)**
   - Generates synthetic minority samples
   - Effective for recall but causes precision collapse

2. **Clustering-Based Oversampling (CBO)**
   - Uses K-Means clustering before oversampling
   - Maintains minority class distribution better than SMOTE

3. **Clustering-Based Undersampling (CBU)**
   - Reduces majority class using cluster centroids
   - Preserves representative samples from majority class

4. **Gaussian Mixture Models (GMM)**
   - Models minority class probability distribution
   - Generates more realistic synthetic samples

## Key Insights

### 1. The Class Weight Paradox
**Critical Discovery:** After using resampling techniques to balance the dataset, applying class weights (giving less weight to the now-balanced minority class) significantly improved performance. This suggests:

- **Resampling makes the minority look bigger than it truly is.** In our case, most fraud samples are synthetic, so the model “sees” large number of fraud cases even though only ~500 are real.  
- **If we leave weights equal, we double-count the minority.** The model treats synthetic frauds as equally important as real ones i.e. it overfits noisy synthetic patterns → high recall but terrible precision.  
- **Lowering the fraud class weight acts like a reality check.** By setting minority class weight very low (e.g., 0.006), we downplay synthetic noise and remind the model: *"Fraud exists, but don’t let the oversampled data trick you into predicting it everywhere."*  
- **Result:** The model becomes conservative — predicting fraud only with strong evidence — which boosts precision while keeping recall at a healthy level.  

### 2. The GMM Illusion  
- **Theoretical Promise:** In principle, a **Gaussian Mixture Model (GMM)** should have elegantly addressed the class imbalance problem by modeling the underlying distribution of both fraud and non-fraud classes. GMMs can generate realistic minority samples by learning cluster-specific covariance structures, thereby preserving the natural variability of fraud patterns.  

- **The Practical Breakdown:** In reality, the synthetic fraud points produced by GMM often **overlapped heavily with normal transactions**, making the decision boundary fuzzier rather than clearer. Instead of improving separability, this overlap confused the model and diluted the fraud signal. To make matters worse, **Cluster-Based Undersampling (CBU)**, while effective in balancing, became too aggressive stripping away a large chunk of the majority class and thus removing crucial context of a "normal transaction".  

- While GMM had the right theoretical machinery, the data's intrinsic overlap plus the information loss from CBU meant the model ended up less discriminative than expected. It highlights a deeper truth: **synthetic balance isn't useful unless the generated data truly respects class boundaries.**  



### 3. Performance Trade-offs
- **Pure Resampling:** High recall (0.85-0.89) but precision collapse (0.03-0.09)
- **Resampling + Class Weights:** Achieves balance between precision (0.68-0.82) and recall (0.67-0.80)
- **Advanced Combinations:** CBU+GMM+Class Weights achieves near-optimal performance

### 4. Sophisticated Preprocessing vs. Complex Algorithms
- **Logistic Regression + Advanced Preprocessing:** F1-score of 0.808 with balanced precision and recall
- **Random Forest (Complex Algorithm):** Slightly higher F1-score (0.827) but at the cost of model complexity
- **Key Insight:** Advanced data preprocessing techniques can achieve performance comparable to complex ensemble methods

## Recommendations

### Primary Recommendation: Sophisticated Preprocessing Approach
**Logistic Regression + CBU + GMM + Class Weights**
- **Strong F1-Score:** 0.808 (only 0.019 behind Random Forest)
- **Balanced Performance:** Precision 0.819, Recall 0.797
- **Model Interpretability:** Maintains explainability of logistic regression
- **Computational Efficiency:** Lower complexity than ensemble methods
- **Business Value:** Demonstrates that advanced data preprocessing can match complex algorithms

### Alternative Solutions by Use Case:

1. **Maximum Performance (if complexity is acceptable):** Random Forest + CBU + GMM (F1: 0.827)
   - Highest performance but at cost of interpretability and complexity
   
2. **Simple Implementation:** Logistic Regression + Class Weights (F1: 0.735)
   - Easy to implement, significant improvement over baseline
   
3. **Balanced Approach:** Logistic Regression + CBO + Class Weights (F1: 0.741)
   - Good performance with moderate preprocessing complexity

### Methods to Avoid:
- Plain resampling without class weights (precision < 0.1)
- Standalone undersampling techniques (CBU alone: F1: 0.068)
- SMOTE without class weight correction (precision: 0.063)

## Technical Implementation

### Project Structure
```bash
credit-fraud-detection/
├── addressing-class-imbalance-via-synthetic-sample-generation.ipynb
├── requirements.txt
└── README.md
```

### Key Dependencies
```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
imblearn
```

## Business Impact

### Cost-Benefit Analysis
- **False Positives:** Customer inconvenience, manual review costs
- **False Negatives:** Direct financial loss, reputation damage
- **Optimal Model:** Random Forest + CBU + GMM balances both concerns

### Real-World Deployment Considerations
1. **Threshold Tuning:** Fine-tune decision boundary based on business priorities
2. **Monitoring:** Continuous performance tracking as fraud patterns evolve
3. **Interpretability:** Use Logistic Regression variant when explanations are required
4. **Scalability:** Random Forest handles large datasets efficiently

## Future Work

1. **Ensemble Methods:** Combine top-performing models for even better results
2. **Deep Learning:** Explore neural networks with custom loss functions
3. **Adversarial Robustness:** Test against evolving fraud techniques

## Conclusion

This comprehensive analysis demonstrates that **sophisticated resampling combined with careful class weight tuning** can achieve performance nearly matching complex ensemble methods. The key insight—that balanced datasets still benefit from class weights—reveals the importance of guiding model behavior even after addressing class imbalance.

The **Logistic Regression + CBU + GMM + Class Weights** approach represents an optimal balance between performance and interpretability, achieving an F1-score of 0.808 while maintaining model explainability - demonstrating that advanced preprocessing techniques can be as powerful as complex algorithmic approaches.
