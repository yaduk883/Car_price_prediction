# Car_price_prediction

Predicting car prices in the US market to help a new entrant understand which design and specification factors drive pricing, and by how much.

## Dataset

205 rows, 26 columns, no missing values or duplicates. Includes car
specifications (engine size, horsepower, dimensions, fuel economy),
categorical attributes (body style, drivetrain, fuel type, engine type), and
the target variable `price`.

## Methodology

1. **Preprocessing**
   - Dropped the `car_ID` identifier.
   - Extracted brand (`CompanyName`) from the free-text `CarName` field and
     corrected spelling inconsistencies in the raw data (e.g. `maxda` →
     `mazda`, `porcshce` → `porsche`, `toyouta` → `toyota`, `vokswagen`/`vw`
     → `volkswagen`).
   - Mapped `doornumber` and `cylindernumber` (e.g. `"four"`) to integers to
     preserve their ordering.
   - One-hot encoded remaining nominal categoricals (`fueltype`,
     `aspiration`, `carbody`, `drivewheel`, `enginelocation`, `enginetype`,
     `fuelsystem`, `CompanyName`) with `drop_first=True`.
   - 80/20 train/test split (`random_state=42`), features scaled with
     `StandardScaler` fit on the training set only.

2. **Models implemented**: Linear Regression, Decision Tree Regressor,
   Random Forest Regressor, Gradient Boosting Regressor, Support Vector
   Regressor — all trained on the identical split for a fair comparison.

3. **Evaluation**: R², MSE, RMSE, MAE on the held-out test set.

4. **Feature importance**: impurity-based importances from the two tree
   ensembles, cross-checked against standardized Linear Regression
   coefficients.

5. **Hyperparameter tuning**: `GridSearchCV` (5-fold CV, optimizing R²) for
   Decision Tree, Random Forest, Gradient Boosting, and SVR.

## Results

### Baseline (default hyperparameters)

| Model | R² | RMSE ($) | MAE ($) |
|---|---|---|---|
| **Random Forest** | **0.958** | **1,826** | **1,286** |
| Gradient Boosting | 0.929 | 2,370 | 1,666 |
| Decision Tree | 0.898 | 2,842 | 2,033 |
| Linear Regression | 0.895 | 2,878 | 1,942 |
| Support Vector Regressor | −0.101 | 9,322 | 5,702 |

### After hyperparameter tuning

| Model | R² | RMSE ($) | MAE ($) |
|---|---|---|---|
| **Random Forest** | **0.958** | **1,825** | **1,288** |
| Gradient Boosting | 0.933 | 2,302 | 1,737 |
| Decision Tree | 0.890 | 2,947 | 2,100 |
| Support Vector Regressor | 0.878 | 3,110 | 1,931 |

**Random Forest is the best model**, both before and after tuning — it
explains ~96% of test-set price variance with a typical error of about
$1,286. Tuning barely moves it because it was already near its practical
ceiling on this dataset; the standout tuning result instead is SVR, which
goes from unusable (negative R²) to competitive (R² 0.878) purely by
increasing `C`, since the default `C=1` is far too small relative to the
$5,000–$45,000 price scale.

### Most significant predictors

`enginesize` and `curbweight` dominate every model, together accounting for
roughly 75–85% of total feature importance, followed by `highwaympg`,
`horsepower`, `carwidth`/`wheelbase`, and specific brand indicators (brand
carries pricing power independent of specifications — e.g. Jaguar, Buick,
Porsche and BMW price well above mechanically comparable cars from
Chevrolet, Dodge, or Plymouth).

## Business Recommendations

1. Engine size and vehicle weight are the biggest levers on price
   positioning — small design changes here move price more than any other
   decision.
2. Fuel economy and physical footprint are secondary but material levers for
   fine-tuning within a target price band.
3. Brand/segment positioning matters independently of specs, so brand
   strategy and product engineering should be planned together when entering
   the US market.

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook CarPrice_Prediction.ipynb
```

Run all cells top to bottom; the notebook is fully self-contained (reads
`CarPrice_Assignment.csv` from the same directory) and reproducible
(`random_state=42` throughout).

## Requirements

- Python 3.9+
- pandas, numpy, scikit-learn, matplotlib, seaborn

- Run on Google Colab - https://colab.research.google.com/drive/1L5oUviLfJdTerUhMT0nFag6jsqTCUpCE?usp=sharing
-  
