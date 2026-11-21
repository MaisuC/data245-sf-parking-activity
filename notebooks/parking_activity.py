#!/usr/bin/env python
# coding: utf-8

# # Forecasting San Francisco Parking Meter Payment Activity
#
# **DATA 245 group project**
# Jietian Chen, Eric Zhao, Alaa Ahmed, Tramy Nguyen
#
# **Question:** Can classical machine learning forecast next-hour meter payment activity better than simple historical baselines?
#
# The proposal originally discussed parking availability. Transactions do not supply occupancy ground truth. The implemented target is **new paid-session starts per eligible single-space meter**, classified using training-only terciles. High activity can reflect rapid turnover and must not be equated with few vacant spaces.
#
# All displayed numbers come from the committed public DataSF observations and the executable Python pipeline. No synthetic observations or invented experiment results are used.

# In[1]:


from pathlib import Path
import sys, json
import pandas as pd
import joblib
from IPython.display import display, Image

ROOT = next(p for p in [Path.cwd(), Path.cwd().parent] if (p / 'src/train.py').exists())
sys.path.insert(0, str(ROOT))
pd.set_option('display.max_columns', 20)
print('Project root:', ROOT)


# ## 1. Sources and sampling
#
# - [DataSF detailed revenue transactions](https://data.sfgov.org/d/imvp-dq3v): NS = new session, AT = additional time, SC = status change.
# - [DataSF parking meter inventory](https://data.sfgov.org/d/8vzz-qzz9): POST_ID joins transactions to blockfaces.
# - Fixed observation period: September 1 through October 31, 2026.
# - Inventory-only seed-245 sample: 16 neighborhoods, one eligible blockface each. Grey, single-space, active M/T, SFMTA on-street meters, 12–35 spaces per blockface.
# - The snapshot is later than the payment period. Meter inventory changes and source coverage are limitations.
# - Published timestamps have no timezone offset. We retain those values without claiming verified local-time alignment.
#
# The aggregate CSV and exact API query manifest are included. Full inventory and per-query caches are excluded from Git to keep the repository compact.

# In[2]:


manifest = json.loads((ROOT / 'data/raw/download_manifest.json').read_text())
display(pd.Series({k: manifest[k] for k in ['start_inclusive','end_exclusive','inventory_rows',
    'selected_meters','selected_blockfaces','source_event_count']}))
coverage = pd.read_csv(ROOT / 'results/block_coverage.csv', dtype={'blockface_id': str})
display(coverage[['blockface_id','neighborhood','supply','reporting_meters',
                  'reporting_meter_fraction','modeling_rows']])


# ## 2. Forecast definition and leakage controls
#
# At the start of hour H, predict the activity class during [H, H+1). Calendar and location are known. Payment-derived inputs use H−1, H−2, H−24, H−168, and a three-hour trailing mean ending at H−1.
#
# `target_pressure = NS starts in H / eligible single-space meters from the inventory snapshot`.
#
# Classes use the training-only 1/3 and 2/3 quantiles. This rate is an activity intensity, **not an occupancy probability** and is not constrained to [0, 1]. All-day-silent blockfaces/dates receive missing status; within covered block-days, unrecorded hourly events receive zero. Coverage filtering is retrospective and cannot distinguish free parking from data outages.
#
# The analytical window is published hours 09:00–18:59, Monday–Saturday. It does not establish actual charging schedules. Rows with missing target/history are excluded. Dates split approximately 60/20/20, with every blockface at the same timestamp in the same partition. The scaler and encoder fit on training data during tuning.

# In[3]:


quality = json.loads((ROOT / 'results/data_quality.json').read_text())
display(pd.DataFrame(quality['split_summary']).T)
print('Training-only class boundaries:', quality['thresholds_training_only'])
print('Sampled / modeled blockfaces:', quality['sampled_blockfaces'], '/', quality['modeled_blockfaces'])
data = pd.read_csv(ROOT / 'data/processed/modeling_data.csv', dtype={'blockface_id': str})
display(data[['target_start','blockface_id','supply','pressure_lag1',
              'pressure_lag24','pressure_lag168','target_pressure','target_class','split']].head())


# ## 3. Exploratory analysis
#
# The following charts summarize the final usable observations. Coverage and historical-feature requirements remove observations, so these patterns describe the analyzed sample rather than the entire city.

# In[4]:


for name in ['sampled_locations','weekday_hour_heatmap','class_distribution','chronological_split']:
    display(Image(filename=str(ROOT / f'results/figures/{name}.png'), width=850))


# ## 4. Reproducible training
#
# `src/train.py` implements the complete workflow. It compares Logistic Regression, KNN, Decision Tree, and Random Forest, three configurations each. Validation macro F1 selects each configuration and the final machine-learning algorithm. Each selected configuration then fits train + validation and receives one chronological test evaluation. Majority, persistence, and blockface/weekday/hour seasonal baselines provide reference points.
#
# Set the switch below to `True` to regenerate all results. The default reuses the completed experiment, keeping a quick notebook walkthrough fast. To redownload source data, separately run `python src/download_data.py` from the repository root.

# In[5]:


RUN_TRAINING = False
if RUN_TRAINING:
    from src.train import main as train_project
    train_project()
else:
    print('Using committed experiment outputs. Set RUN_TRAINING=True to rerun.')


# In[6]:


tuning = pd.read_csv(ROOT / 'results/validation_search.csv')
display(tuning.sort_values(['model','validation_macro_f1'], ascending=[True,False]))
metrics = pd.read_csv(ROOT / 'results/test_metrics.csv')
display(metrics.round(4))
display(Image(filename=str(ROOT / 'results/figures/model_comparison.png'), width=900))


# ## 5. Feature ablation and error analysis
#
# These are descriptive analyses with the already selected hyperparameters. They do not retune the final model against the test set. Adding supply separately to block identity may add little because supply is fixed within a blockface. The random-split diagnostic uses different test observations and cannot isolate leakage or prove that random splitting always increases scores.

# In[7]:


display(pd.read_csv(ROOT / 'results/ablation_metrics.csv').round(4))
display(pd.read_csv(ROOT / 'results/error_analysis.csv').round(4))
uncertainty = json.loads((ROOT / 'results/uncertainty.json').read_text())
print(json.dumps(uncertainty, indent=2))
for name in ['confusion_matrix','feature_importance']:
    display(Image(filename=str(ROOT / f'results/figures/{name}.png'), width=850))


# ## 6. Saved-model demonstration
#
# This loads the persisted preprocessing pipeline and classifier using actual held-out features. It does not fetch live data, search for vacant parking spaces, or retrain the model. Only load trusted joblib/pickle files.

# In[8]:


bundle = joblib.load(ROOT / 'models/parking_activity_model.joblib')
example = pd.read_json(ROOT / 'models/example_input.json', dtype={'blockface_id': str})
predicted = int(bundle['pipeline'].predict(example[bundle['features']])[0])
probabilities = bundle['pipeline'].predict_proba(example[bundle['features']])[0]
print('Validation-selected machine-learning model:', bundle['selected_model'])
print('Prediction:', bundle['class_names'][predicted])
display(pd.Series(probabilities, index=bundle['class_names'], name='Predicted probability'))
print('Saved model bytes:', (ROOT / 'models/parking_activity_model.joblib').stat().st_size)


# ## 7. Findings and limits
#
# - The seasonal baseline is competitive with the more flexible algorithms. The strongest test macro F1 belongs to the seasonal baseline; the validation-selected ML model is KNN. A claim that ML outperforms the seasonal rule is unsupported.
# - Historical inputs improve the selected ML model over its calendar/location-only ablation on this held-out period. This comparison describes the sample, not a causal effect.
# - Four originally sampled blockfaces have no events in the period. Twelve blockfaces contribute modeling rows, and some have sparse meter reporting. The inventory denominator does not establish actual operational supply during September–October.
# - No Monday rows survive the required 24-hour history after entirely silent Sundays. The final usable sample covers Tuesday–Saturday and does not validate Monday forecasts.
# - The test covers eight observed dates. The paired-day bootstrap is descriptive and does not establish citywide or seasonal generalization.
# - Paid-session starts cannot validate actual occupancy or parking availability. No measured reduction in congestion, emissions, or cruising is claimed.
# - Evaluation uses one-hour rolling forecasts with newly observed preceding-hour activity. Immediate aggregate availability is an assumption; API ingestion latency has not been measured.
#
# For the original proposal and the eventual report, describe the scope refinement explicitly. Ground-truth occupancy or a validated occupancy reconstruction is future work.

# ## 8. Team and AI disclosure
#
# Team: Jietian Chen, Eric Zhao, Alaa Ahmed, Tramy Nguyen. Individual CRediT roles must reflect work each person actually completes and confirms. See `docs/TEAM_REVIEW.md` for a proposed review division; it is not a record of completed contributions.
#
# OpenAI Codex assisted with implementation, data queries, experiment execution, documentation, and notebook generation. Team members must review and understand the artifacts before submission. No historical scrum meetings or pair-programming sessions are claimed. No leaderboard position is applicable because these are direct city datasets, not competition submissions.
