# Delhivery Last-Mile Delay Intelligence Pipeline

This project is an end-to-end machine-learning pipeline for last-mile delivery operations. Using the supplied synthetic Delhivery shipment data, it answers three practical questions:

- How many hours is a parcel likely to take from dispatch?
- Is the parcel likely to miss its promised delivery time?
- Which hubs need a standard response, closer monitoring, or escalation?

The pipeline cleans the raw shipment exports, builds shared features, trains regression and classification models, clusters hubs by their operating behaviour, and produces an action list for new parcels.

## Project workflow

1. **Part A — data preparation and EDA:** cleans the shipment data, handles missing values and scanner-fault weights, creates model features, and measures the main delay patterns.
2. **Part B — regression:** compares several models for predicting delivery hours.
3. **Part C — classification:** predicts whether a parcel will miss its delivery promise and selects an alert threshold that favours recall.
4. **Part D — segmentation and scoring:** groups the hubs into operating segments, packages the final classifier, and scores new shipments into an operational action list.

## How to run

Run these commands from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 generate_data.py
python3 part_a_clean_eda.py
python3 part_b_regression.py
python3 part_c_classification.py
python3 part_d_segments.py
python3 package_model.py
python3 predict_delay.py
```

`package_model.py` rebuilds `models/delay_model.pkl` and the supporting model files. The saved pickle should be loaded only with the scikit-learn version pinned in `requirements.txt`. If the installed version is different, rerun `package_model.py` before scoring new shipments.

## Data findings

The first pass through the data made the main operational pattern quite clear: delays rise with tighter promises, longer routes, and stretched hubs.

- **Express** has the highest delay rate at **0.4897**. Standard is **0.2056** and Economy is **0.0857**.
- Delay rate increases from **0.0400** at Tier 1 hubs to **0.2150** at Tier 2 and **0.4400** at Tier 3.
- Distance is a strong operational risk factor. The delay rate is **0.0219** for 0–200 km, **0.1368** for 201–400 km, **0.4364** for 401–600 km, and **0.8523** above 600 km.
- The strongest positive correlations with delivery hours are promised hours (**0.8372**), distance km (**0.4013**), and number of stops (**0.3247**).
- Declared value was log-transformed because it is right-skewed. Its filled mean and median are **1533.0542** and **1121.5000** INR, while the log-transformed mean and median are much closer at **7.0277** and **7.0224**.

## Model results (Regression)

For delivery-hour prediction, the mean baseline has a validation R-squared of **-0.0212**, so it adds no useful planning value. Linear regression performed best on the validation set, with RMSE **4.8912** hours and R-squared **0.9313**. Its final test R-squared is **0.9493**.

The decision-tree diagnostic also shows why model complexity needs to be controlled. At depth 1, both train and validation errors are high (**11.6334** and **12.1324** RMSE), which indicates high bias. An unlimited tree fits the training data perfectly (**0.0000** RMSE) but reaches **7.7890** RMSE on validation, showing overfitting. Depth 6 is a more balanced tree, with validation RMSE **6.2796**.

For gradient boosting, a learning rate of **0.1000** performed best with validation R-squared **0.9306**. A rate of **0.0100** learned too slowly (**0.7127**), while **1.0000** was too aggressive (**0.8440**). Five-fold cross-validation slightly favoured gradient boosting (**0.9352**) over linear regression (**0.9261**), but linear regression remained the selected model because it won the validation comparison used for model selection.

## Model results (Classification)

The all-on-time baseline gives an accuracy of **0.7667**, but it catches none of the late parcels. This is why accuracy alone is misleading for this problem.

The selected model is a random forest using the seven strongest features. The bookkeeping/noise columns contribute a combined **0.2253** of forest importance, but every one of them ranks outside the top seven by F-score. The seven-feature model reaches validation F1 **0.8333**, compared with **0.6988** for five features and **0.8000** when all features are included.

Hyperparameter tuning did not improve the result on the validation set: grid search scored **0.7750** F1 and randomized search **0.7901**, while the untuned forest scored **0.8333**. With a relatively small validation set, the simpler model was the more reliable choice here.

The final alert threshold is **0.5000**. On validation, it gives precision **0.7955** and recall **0.8333**. On the untouched test set, precision is **0.6889**, recall is **0.7561**, and F1 is **0.7209**. The test result should be treated carefully because it contains only 41 delayed parcels.

## Operations recommendation

The Escalate segment contains HUB13 (**0.4600** delay rate), HUB15 (**0.3600**), HUB17 (**0.4600**), HUB18 (**0.3800**), HUB19 (**0.6200**), HUB20 (**0.4800**), HUB21 (**0.4000**), HUB23 (**0.4400**), and HUB24 (**0.5400**). A flagged parcel at one of these hubs should be sent to the hub manager; a flagged parcel at another segment should receive a proactive customer call.

The hub-tier labels do not always describe actual operating behaviour. HUB07 is labelled Tier 1 but belongs in Watch. HUB13 and HUB15 are Tier 2 but belong in Escalate. HUB22 is labelled Tier 3 but belongs in Watch. These four mismatches show why operational decisions should use the segment as well as the assigned hub tier.

The final action list contains **48** new shipments. It flags **8** parcels: **6** go to a hub manager and **2** receive a proactive customer call. At the selected threshold, the test-set precision of **0.6889** means roughly seven out of ten flagged parcels are genuinely delayed, while recall of **0.7561** means roughly eight out of ten delayed parcels are caught.

I kept three hub segments even though the separation ratio rises from **5.3928** at K=3 to **6.3047** at K=4 and **6.8866** at K=5. K=3 gives practical groups of 9-8-7 hubs. The larger K values create clusters with fewer than six hubs, which is too small to justify a separate operations playbook.

## Production plan

- **Accuracy:** use regression R-squared **0.9493** and classification precision **0.6889**, recall **0.7561**, and F1 **0.7209** as initial release baselines.
- **Reliability:** the model reload check scores 180 test rows and reports a maximum probability difference of **0.0000**.
- **Reproducibility:** the generator, pinned dependencies, fixed random states, and committed CSV outputs make the workflow repeatable offline.
- **Consistency:** the same `clean_shipments` and `build_features` functions are used for training and scoring. The feature order, imputation values, and threshold are stored alongside the model.
- **Maintainability and monitoring:** each stage writes a CSV output, so the data checks, metrics, feature scores, and segmentation results can be reviewed independently.

I would monitor weekly delayed share, alert precision, and alert recall after real delivery outcomes become available. The initial review triggers are a delayed share above **0.2317**, precision below **0.6889**, or recall below **0.7561**. Retraining should happen quarterly or sooner if a trigger persists. During the 15–28 September sale window, performance should be compared with similar sale periods rather than ordinary weeks; otherwise, expected sale-event behaviour could be mistaken for data drift.

## Limitations

- The dataset is synthetic, so the results are useful for demonstrating the pipeline but not for making claims about real Delhivery operations.
- The classification test set has only 41 delayed parcels, so the reported precision and recall can move noticeably with a different sample.
- The threshold was selected from a single validation split and should be checked against time-based validation before deployment.
- Hub segments are based on historical averages. They need to be refreshed as lane mix, volume, capacity, and weather patterns change.
