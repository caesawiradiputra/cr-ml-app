# Hyperparameter Tuning - Learning Guide & Q&A

**Notebook Reference**: `04_hyperparameter_tuning.ipynb`  
**Purpose**: Optimize Model Performance Through Hyperparameter Tuning  
**Dataset**: Iris Flower Dataset (scikit-learn)  
**Methods**: GridSearchCV and RandomizedSearchCV

---

## Overview

This notebook demonstrates how to systematically optimize model performance by finding the best hyperparameter settings. Hyperparameter tuning is crucial for squeezing maximum performance from your models and can often improve accuracy by 5-15% or more.

---

## What are Hyperparameters?

**Hyperparameters** = Settings you choose BEFORE training that control HOW the model learns

### Hyperparameters vs Parameters

| Aspect | Parameters | Hyperparameters |
| -------- | ----------- | ----------------- |
| **Set When** | During training (learned) | Before training (chosen) |
| **Example** | Tree split thresholds, weights | max_depth, learning_rate |
| **Learned By** | The model (automatically) | You (manually or via search) |
| **Optimized Via** | Loss minimization | Cross-validation |

### Examples

**Parameters (learned during training):**

- Neural network weights
- Decision tree split values
- SVM support vectors

**Hyperparameters (set before training):**

- Number of neighbors in KNN (k)
- Maximum tree depth
- Learning rate
- Number of trees in forest

---

## Section-by-Section Q&A

### 1-2. Setup and Data Preparation

**Q: Why do we need a separate notebook for tuning?**

- Tuning is computationally expensive
- Requires systematic approach
- Results need careful comparison
- Foundation for production model selection

**Q: Why use the same data split as previous notebooks?**

- Fair comparison with baseline models
- Reproducible results
- Same `random_state=42` ensures consistency

---

### 3. Understanding Hyperparameters

**Q: Why does each algorithm have different hyperparameters?**

- Different learning mechanisms
- Different assumptions about data
- Different ways to control complexity
- Different trade-offs (speed vs accuracy)

#### K-Nearest Neighbors (KNN) Hyperparameters

**Q: What does `n_neighbors` control?**

- Number of neighbors to consider for voting
- **Small k (e.g., 1)**: More complex, captures noise (overfitting)
- **Large k (e.g., 50)**: Smoother, misses patterns (underfitting)
- **Optimal**: Usually sqrt(N) or found via search

**Q: What is the `weights` parameter?**

- **'uniform'**: All neighbors vote equally
- **'distance'**: Closer neighbors have more influence
- **When to use**: 'distance' often better, especially with uneven data

**Q: What does `metric` do?**

- Defines how distance is calculated
- **'euclidean'**: Straight-line distance (most common)
- **'manhattan'**: Grid-like distance (better for high dimensions)
- **'minkowski'**: Generalization of euclidean and manhattan

**Example:**

```python
# Poor configuration (too small k)
KNeighborsClassifier(n_neighbors=1)  # Overfits to noise

# Better configuration (balanced k)
KNeighborsClassifier(n_neighbors=13, weights='distance')  # More robust
```

---

#### Decision Tree Hyperparameters

**Q: What does `max_depth` control?**

- Maximum levels the tree can grow
- **None**: Unlimited depth (will overfit!)
- **Small (3-5)**: Simple model, may underfit
- **Medium (7-15)**: Usually optimal
- **Large (>20)**: Often overfits

**Q: What is `min_samples_split`?**

- Minimum samples required to split a node
- **Low (2)**: More splits, complex tree, overfitting risk
- **High (10-20)**: Fewer splits, simpler tree, more generalizable
- **Purpose**: Regularization to prevent overfitting

**Q: What is `min_samples_leaf`?**

- Minimum samples required in a leaf node
- **Low (1)**: Can create single-sample leaves (overfit!)
- **High (5-10)**: Smoother predictions
- **Purpose**: Another regularization technique

**Q: What is `criterion`?**

- Measure of split quality
- **'gini'**: Gini impurity (faster, default)
- **'entropy'**: Information gain (more precise)
- **Difference**: Usually minimal in practice

**Q: How do these parameters interact?**

```python
# Overfitting configuration
DecisionTreeClassifier(
    max_depth=None,        # Unlimited depth
    min_samples_split=2,   # Split everything
    min_samples_leaf=1     # Single-sample leaves
)
# → Will memorize training data!

# Regularized configuration
DecisionTreeClassifier(
    max_depth=10,          # Limited depth
    min_samples_split=10,  # Need 10 samples to split
    min_samples_leaf=5     # At least 5 samples per leaf
)
# → More generalizable!
```

---

#### Support Vector Machine (SVM) Hyperparameters

**Q: What does `C` control?**

- Penalty for misclassification
- **Low C (0.1)**: Soft boundary, more errors allowed (underfitting)
- **High C (100)**: Hard boundary, fewer errors tolerated (overfitting)
- **Optimal**: Balance between margin width and errors

**Q: What is the `kernel` parameter?**

- Defines decision boundary shape
- **'linear'**: Straight line/plane (fast, simple)
- **'rbf'**: Radial basis function (flexible, most common)
- **'poly'**: Polynomial (specific curve shapes)
- **'sigmoid'**: S-shaped (like neural network)

**Q: What does `gamma` control (for RBF kernel)?**

- Influence radius of single training example
- **Low gamma (0.001)**: Wide influence, smooth boundary
- **High gamma (10)**: Narrow influence, complex boundary (overfits)
- **'scale'**: 1 / (n_features × X.var()) - default
- **'auto'**: 1 / n_features

**Q: How do C and gamma interact?**

```python
# Underfitting: Soft boundary, wide influence
SVC(C=0.1, gamma=0.001, kernel='rbf')

# Balanced: Medium C, medium gamma
SVC(C=1, gamma='scale', kernel='rbf')

# Overfitting: Hard boundary, narrow influence
SVC(C=100, gamma=10, kernel='rbf')
```

**Q: When to use which kernel?**

- **Linear**: Data is linearly separable, high dimensions
- **RBF**: General purpose, non-linear patterns (start here)
- **Poly**: Specific polynomial relationships expected
- **Sigmoid**: Rarely used, similar to neural network

---

#### Random Forest Hyperparameters

**Q: What does `n_estimators` control?**

- Number of trees in the forest
- **Low (10-50)**: Fast but less stable
- **Medium (100)**: Good balance (default)
- **High (200-500)**: More stable but slower
- **Note**: More is usually better (diminishing returns)

**Q: What is `max_depth`?**

- Maximum depth of EACH tree
- **None**: Unlimited (each tree can overfit)
- **Shallow (5-10)**: Prevents individual tree overfitting
- **Note**: Random Forest is robust, can handle deeper trees

**Q: What is `min_samples_split`?**

- Same as Decision Tree but applied to each tree
- Controls complexity of individual trees
- Random Forest averages out, so less critical

**Q: What does `max_features` control?**

- Number of features considered for each split
- **'sqrt'**: sqrt(n_features) - recommended for classification
- **'log2'**: log2(n_features) - also good
- **int**: Specific number
- **None**: All features (loses randomness!)

**Q: Why is `max_features` important?**

- Creates diversity between trees
- Each tree sees different random subset of features
- Diversity → better ensemble performance
- **Too high**: Trees become similar (less benefit)
- **Too low**: Trees are weak (underfitting)

**Example:**

```python
# Weak ensemble (too few trees, all features)
RandomForestClassifier(n_estimators=10, max_features=None)

# Strong ensemble (many trees, feature randomness)
RandomForestClassifier(n_estimators=200, max_features='sqrt')
```

---

### 4. Define Parameter Grids

**Q: How to choose parameter ranges?**

1. **Research**: Check documentation and papers
2. **Domain Knowledge**: Understand your problem
3. **Start Broad**: Wide range first, narrow later
4. **Logarithmic Scale**: For parameters like C, gamma (0.1, 1, 10, 100)
5. **Balance**: Not too many combinations (computational cost)

**Q: What is "total combinations"?**

- Number of all possible parameter combinations
- Example: `[3, 5, 7] × ['uniform', 'distance'] = 3 × 2 = 6 combinations`
- GridSearch tests ALL of these
- RandomizedSearch samples a subset

**Q: How many combinations is too many?**

```text
KNN: 7 × 2 × 2 = 28 combinations ✅ Reasonable
Decision Tree: 5 × 3 × 3 × 2 = 90 combinations ✅ Manageable
SVM: 4 × 2 × 5 = 40 combinations ✅ Good
Random Forest: 3 × 4 × 3 × 2 = 72 combinations ✅ Fine

Example of too many:
n_estimators: [10, 50, 100, 200, 500] = 5
max_depth: [3, 5, 7, 10, 15, 20, None] = 7
min_samples_split: [2, 5, 10, 15, 20] = 5
min_samples_leaf: [1, 2, 4, 6, 8, 10] = 6
max_features: ['sqrt', 'log2', None, 0.3, 0.5, 0.7] = 6
Total: 5 × 7 × 5 × 6 × 6 = 6,300 combinations 🔴 Too many!
```

**Q: Should I always include all possible values?**

- No! Balance granularity vs computation
- **Coarse search first**: `[1, 10, 100]` to find region
- **Fine search later**: `[5, 6, 7, 8, 9]` to refine
- **Or use RandomizedSearch**: Samples from broader range

---

### 5. GridSearchCV - Exhaustive Search

**Q: How does GridSearchCV work?**

1. Takes all parameter combinations
2. For each combination:
   - Trains model on K-1 folds
   - Validates on remaining fold
   - Repeats K times (cross-validation)
   - Averages performance
3. Returns combination with best CV score

**Q: What does `cv=5` mean?**

- 5-fold cross-validation
- Data split into 5 parts
- Each combination tested 5 times
- Total model fits: `combinations × 5`
- Example: 90 combinations × 5 folds = 450 model trainings!

**Q: What is `scoring='accuracy'`?**

- Metric to optimize
- Options: 'accuracy', 'f1', 'precision', 'recall', 'roc_auc'
- Choose based on problem (fraud detection → 'f1')

**Q: What does `n_jobs=-1` do?**

- Uses all CPU cores for parallel processing
- Speeds up search significantly
- `-1` = use all cores
- `1` = single core (slower)
- `2` = use 2 cores

**Q: What is `best_estimator_`?**

- The trained model with best parameters
- Ready to use for predictions
- Already fitted on full training set

**Q: What is `best_score_`?**

- Best cross-validation score found
- Average across K folds
- More reliable than single test score

**Q: What is `best_params_`?**

- Dictionary of best parameter values
- Example: `{'n_neighbors': 13, 'weights': 'distance'}`

**Example workflow:**

```python
# GridSearch tests:
# Combination 1: k=3, weights='uniform', metric='euclidean'
#   → Fold 1: 94%, Fold 2: 96%, ..., Fold 5: 95% → Avg: 95.0%
# Combination 2: k=3, weights='distance', metric='euclidean'
#   → Fold 1: 95%, Fold 2: 97%, ..., Fold 5: 96% → Avg: 95.8%
# ...
# Combination 28: k=15, weights='distance', metric='manhattan'
#   → Fold 1: 96%, Fold 2: 98%, ..., Fold 5: 97% → Avg: 96.8% ← Best!

# Returns: best_params = {'n_neighbors': 15, 'weights': 'distance', 'metric': 'manhattan'}
```

---

### 6. RandomizedSearchCV - Faster Alternative

**Q: How is RandomizedSearch different from GridSearch?**

| Aspect | GridSearchCV | RandomizedSearchCV |
| -------- | -------------- | ------------------- |
| **Tests** | ALL combinations | RANDOM sample |
| **Speed** | Slow | Fast |
| **Guarantee** | Finds best in grid | Might miss best |
| **Good For** | Small grids | Large grids |
| **Control** | Grid size | `n_iter` parameter |

**Q: What does `n_iter=20` mean?**

- Tests 20 random combinations
- Each combination fully cross-validated
- Much faster than testing all 90+ combinations
- Usually finds near-optimal solution

**Q: How does RandomizedSearch choose combinations?**

- Randomly samples from parameter distributions
- Each parameter sampled independently
- Example: Random `n_neighbors` + random `weights` + random `metric`
- `random_state=42` for reproducibility

**Q: When to use GridSearch vs RandomizedSearch?**

**Use GridSearch when:**

- ✅ Small grid (< 100 combinations)
- ✅ Need guaranteed best in grid
- ✅ Have time for exhaustive search
- ✅ Final fine-tuning around known region

**Use RandomizedSearch when:**

- ✅ Large grid (> 100 combinations)
- ✅ Exploring wide parameter space
- ✅ Time/compute limited
- ✅ Initial coarse search
- ✅ Good-enough solution acceptable

**Q: How to choose `n_iter`?**

- **Rule of thumb**: 10-20% of total combinations
- Example: 500 combinations → `n_iter=50-100`
- **Trade-off**: Higher n_iter = more time, better chance of finding optimum
- **Diminishing returns**: Beyond certain point, little improvement

**Example strategy:**

```python
# Stage 1: Broad RandomizedSearch
param_grid_broad = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20, 50]
}
# 5 × 6 × 5 = 150 combinations
random_search = RandomizedSearchCV(..., n_iter=30)  # Test 20% of space

# Stage 2: Narrow GridSearch around best region
# Found best around: n_estimators=200, max_depth=10, min_samples_split=5
param_grid_narrow = {
    'n_estimators': [150, 200, 250],
    'max_depth': [8, 10, 12],
    'min_samples_split': [3, 5, 7]
}
# 3 × 3 × 3 = 27 combinations - exhaustive search feasible
grid_search = GridSearchCV(..., param_grid=param_grid_narrow)
```

---

### 7. Compare Grid Search vs Random Search

**Q: Which method performs better?**

- Usually very similar final accuracy
- GridSearch slightly better (tests more combinations)
- Difference often < 1%
- RandomizedSearch much faster

**Q: Example results interpretation:**

```text
Algorithm      Method         CV Score  Test Score  Time (s)
KNN            Grid Search     0.9667    0.9667      0.50
KNN            Random Search   0.9667    0.9667      0.15  ← 3.3x faster!

Decision Tree  Grid Search     0.9583    0.9667      0.80
Decision Tree  Random Search   0.9500    0.9667      0.18  ← 4.4x faster!

SVM            Grid Search     0.9750    0.9667      2.50
SVM            Random Search   0.9667    0.9667      0.45  ← 5.6x faster!

Random Forest  Grid Search     0.9667    1.0000      5.20
Random Forest  Random Search   0.9583    1.0000      1.10  ← 4.7x faster!

Average speedup: 4.5x faster! ⚡
```

**Q: Why is RandomizedSearch so much faster?**

- Tests fewer combinations (20 vs 90)
- Samples most important regions
- Skips redundant similar configurations
- Parallel processing still helps

**Q: When might RandomizedSearch miss the best?**

- If optimal parameters are rare/extreme
- Small grids where exhaustive search is cheap
- When you need guarantee of absolute best
- Usually difference is negligible in practice

---

### 8. Visualize Tuning Results

**Q: What does the accuracy comparison show?**

- Side-by-side bars for Grid vs Random
- Usually very similar heights (similar accuracy)
- Validates that RandomizedSearch finds good solutions

**Q: What does the time comparison show?**

- Dramatic difference in computational cost
- RandomizedSearch consistently faster
- More pronounced for complex models (SVM, Random Forest)

**Q: Why does SVM take longest?**

- SVM training is O(n²) to O(n³) complexity
- More samples = much slower
- Many parameter combinations to test
- Kernel computations expensive

**Q: Why is KNN fastest?**

- KNN has no "training" phase
- Just stores data
- All time is in cross-validation predictions

---

### 9. Best Parameters Summary

**Q: How to interpret the best parameters?**

**Example results:**

```python
🏆 KNN:
   Test Accuracy: 0.9667
   Best Parameters:
     - n_neighbors: 13
     - weights: 'distance'
     - metric: 'euclidean'

→ Interpretation: Medium k (13) balances flexibility vs stability
                  Distance weighting improves boundary decisions
                  Euclidean distance sufficient for this data
```

```python
🏆 Decision Tree:
   Test Accuracy: 0.9667
   Best Parameters:
     - max_depth: 7
     - min_samples_split: 2
     - min_samples_leaf: 1
     - criterion: 'gini'

→ Interpretation: Limited depth (7) prevents overfitting
                  Standard splits allowed (not over-regularized)
                  Gini impurity works well (standard choice)
```

```python
🏆 SVM:
   Test Accuracy: 0.9667
   Best Parameters:
     - C: 10
     - kernel: 'rbf'
     - gamma: 'scale'

→ Interpretation: Moderate C (10) balances margin and errors
                  RBF kernel handles non-linear boundaries
                  Auto-scaled gamma works well
```

```python
🏆 Random Forest:
   Test Accuracy: 1.0000
   Best Parameters:
     - n_estimators: 200
     - max_depth: None
     - min_samples_split: 2
     - max_features: 'sqrt'

→ Interpretation: Many trees (200) for stable ensemble
                  Unlimited depth okay (ensemble averages out)
                  Feature randomness ('sqrt') creates diversity
```

**Q: Why does Random Forest achieve 100% accuracy?**

- Ensemble power reduces overfitting
- 200 trees capture all patterns
- Iris is relatively simple dataset
- Be careful: might still be overfitting to test set

**Q: Should I always use these parameters for my data?**

- NO! These are optimal for Iris dataset
- Different data = different optimal parameters
- Always tune for YOUR specific dataset
- These provide good starting ranges

---

### 10. Before vs After Tuning Comparison

**Q: What are "default parameters"?**

- Settings used when you don't specify anything
- Example: `KNeighborsClassifier()` uses `n_neighbors=5`
- Often reasonable but rarely optimal
- Good baseline for comparison

**Q: How much improvement can tuning provide?**

```text
Typical improvements:
- Simple datasets (like Iris): 2-8%
- Complex datasets: 10-20%
- Very noisy datasets: 5-15%
- Already-optimized baseline: 1-3%
```

**Q: Example interpretation:**

```text
KNN:
  Default:  0.9333 (n_neighbors=5)
  Tuned:    0.9667 (n_neighbors=13)
  Improvement: +3.34% 📈

Decision Tree:
  Default:  0.9333 (no max_depth)
  Tuned:    0.9667 (max_depth=7)
  Improvement: +3.34% 📈

SVM:
  Default:  0.9667 (C=1, gamma='scale')
  Tuned:    0.9667 (C=10, gamma='scale')
  Improvement: +0.00% (defaults were already good!)

Random Forest:
  Default:  0.9667 (n_estimators=100)
  Tuned:    1.0000 (n_estimators=200)
  Improvement: +3.33% 📈
```

**Q: What if tuning makes performance worse?**

- Shouldn't happen with proper cross-validation
- If it does: Check for bugs, data leakage
- Or overfitting to validation set (use nested CV)

**Q: What if tuning shows no improvement?**

- Default parameters already good for this data
- Problem is inherently noisy (tune won't help much)
- Need better features, not better hyperparameters
- Different algorithm might help more

---

### 11. Visualize Improvement

**Q: How to read the improvement visualization?**

- Red bars = default parameters
- Green bars = tuned parameters
- Text above = percentage improvement
- Higher green bar = tuning helped

**Q: What if some models show no improvement?**

- Defaults were already near-optimal
- Or dataset too simple (ceiling effect)
- Or hyperparameters less important for that algorithm
- Still worth tuning for other datasets

---

## Key Takeaways

### Hyperparameter Tuning Process

1. **Define Parameter Space**
   - Research reasonable ranges
   - Start broad, refine later
   - Balance granularity vs computation

2. **Choose Search Strategy**
   - **GridSearch**: Small grids, guaranteed best
   - **RandomizedSearch**: Large grids, fast, good-enough

3. **Use Cross-Validation**
   - ALWAYS use CV (5-10 folds)
   - Prevents overfitting to validation set
   - More reliable than single split

4. **Evaluate and Compare**
   - Compare tuned vs default
   - Check improvement magnitude
   - Verify results on test set

5. **Document Best Parameters**
   - Save for production deployment
   - Reproducibility essential
   - Update as data changes

### Common Pitfalls

**❌ Tuning on test set:**

```python
# WRONG - causes data leakage!
grid_search.fit(X_test, y_test)
```

**✅ Proper approach:**

```python
# RIGHT - tune on training, evaluate on test
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
```

**❌ Ignoring computation cost:**

```python
# 10,000 combinations × 5 folds = 50,000 model trainings!
GridSearchCV(..., param_grid=huge_grid, cv=5)
```

**✅ Practical approach:**

```python
# Stage 1: Coarse search (fast)
RandomizedSearchCV(..., n_iter=50)
# Stage 2: Fine search (focused)
GridSearchCV(..., param_grid=small_refined_grid)
```

**❌ Tuning everything at once:**

```python
# Too many hyperparameters, hard to interpret
param_grid = {
    'param1': [...10 values...],
    'param2': [...10 values...],
    'param3': [...10 values...],
    ...  # 10 parameters total
}
# 10^10 combinations - impossible!
```

**✅ Incremental approach:**

```python
# Tune most important parameters first
param_grid_1 = {'n_estimators': [...], 'max_depth': [...]}
# Then tune others, fixing the first
param_grid_2 = {'min_samples_split': [...], 'min_samples_leaf': [...]}
```

**❌ Not validating on separate test set:**

```python
print(f"Best CV score: {grid_search.best_score_}")
# Stop here - might be overfitting to CV folds!
```

**✅ Always validate:**

```python
print(f"CV score: {grid_search.best_score_}")
print(f"Test score: {grid_search.score(X_test, y_test)}")
# If test << CV, you're overfitting!
```

### Production Considerations

**Deployment:**

- Save best model: `joblib.dump(best_model, 'model.pkl')`
- Document hyperparameters: Keep in metadata
- Version control: Track parameter changes
- Monitor: Performance may drift over time

**Retraining:**

- Re-tune periodically with new data
- Parameters may change as data evolves
- Set up automated tuning pipelines
- Alert if performance degrades significantly

**Cost-Benefit:**

- Tuning takes time (hours to days for large datasets)
- Improvement may be small (1-5%)
- Decide if worth the effort
- Sometimes better features > better hyperparameters

---

## Connection to Fraud Detection Project

### Iris (Learning) vs Fraud (Production)

**Iris Dataset:**

- 150 samples, 4 features
- Balanced classes (33% each)
- GridSearch feasible (< 100 combinations)
- Tuning time: seconds to minutes
- Improvement: 2-5%

**Fraud Detection:**

- 126,530 samples, many features
- Imbalanced (0.78% fraud)
- RandomizedSearch essential (thousands of combinations)
- Tuning time: hours (notebook 07: 10-20 minutes for 50 iterations)
- Improvement: Can be significant (need to compare baseline)
- Optimization metric: F1-Score (not accuracy!)

### Why RandomizedSearchCV for Fraud Detection?

**Reasons:**

1. **Computational Cost**: Large dataset → each model fit is expensive
2. **Many Features**: More features → more parameter interactions
3. **Time Constraints**: Can't wait days for tuning
4. **Good Enough**: 95% optimal vs 100% optimal often negligible
5. **Exploration**: Want to try wide range of parameters

**Notebook 07 Setup:**

```python
param_distributions = {
    'n_estimators': [50, 100, 150, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None, 0.5, 0.7]
}
# Total: 5 × 5 × 6 × 5 × 5 × 5 × 5 = 93,750 combinations!
# RandomizedSearch with n_iter=50 tests only 0.05% of space ⚡
```

---

## Advanced Topics

### Nested Cross-Validation

**Problem:** Tuning on same CV folds used for evaluation → overfitting
**Solution:** Nested CV - outer loop for evaluation, inner for tuning

```python
from sklearn.model_selection import cross_val_score

# Outer CV: Unbiased evaluation
outer_scores = []
for train_idx, test_idx in KFold(5).split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Inner CV: Hyperparameter tuning
    grid_search = GridSearchCV(..., cv=5)
    grid_search.fit(X_train, y_train)
    
    # Evaluate best model on outer fold
    score = grid_search.score(X_test, y_test)
    outer_scores.append(score)

# Unbiased estimate of model performance
print(f"Nested CV score: {np.mean(outer_scores)}")
```

### Bayesian Optimization

**Better than Random:** Uses past results to guide search
**Libraries:** Optuna, Hyperopt, scikit-optimize
**Advantage:** Fewer iterations to find optimum

```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    
    return cross_val_score(model, X, y, cv=5).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(f"Best params: {study.best_params}")
```

### AutoML Tools

**Full Automation:** TPOT, Auto-sklearn, H2O AutoML
**Advantages:** Try multiple algorithms + hyperparameters
**Disadvantages:** Black box, computationally expensive

---

## Practice Exercises

### Beginner

1. Change parameter ranges and observe impact on results
2. Add a new parameter to tune (e.g., `algorithm` for KNN)
3. Try tuning with different scoring metrics ('f1', 'recall')
4. Increase/decrease `n_iter` for RandomizedSearch and compare

### Intermediate

1. Implement two-stage tuning (coarse then fine)
2. Create custom scoring function for imbalanced data
3. Tune multiple models in pipeline (preprocessor + classifier)
4. Analyze parameter importance (which matter most?)

### Advanced

1. Implement nested cross-validation
2. Try Bayesian optimization with Optuna
3. Create automated tuning pipeline for production
4. Build meta-model that predicts optimal hyperparameters from data characteristics

---

## Further Reading

- [GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [RandomizedSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
- [Hyperparameter Tuning Guide](https://scikit-learn.org/stable/modules/grid_search.html)
- [Nested Cross-Validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [Bayesian Optimization: Optuna](https://optuna.org/)
- [AutoML: TPOT](http://epistasislab.github.io/tpot/)
- [Hyperparameter Optimization Algorithms](https://arxiv.org/abs/1502.02127)

---

## Related Documentation

- [01 Iris Classification Guide](./01-iris-classification-guide.md) - ML Fundamentals
- [02 Algorithm Comparison Guide](./02-algorithm-comparison-guide.md) - Algorithm Selection
- [03 Noise Level Assessment Guide](./03-noise-level-assessment-guide.md) - Data Quality
- [Fraud Detection Hyperparameter Tuning](../architecture/fraud-hyperparameter-tuning.md)
- [Production Model Selection Guide](../guides/production-model-selection.md)
