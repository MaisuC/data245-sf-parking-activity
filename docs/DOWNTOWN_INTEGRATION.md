# Downtown extension: will a space have no active payment when I arrive?

This extension joins two pieces of DATA 245 work:

- **This repository:** SF payment-data pipeline, calendar and history features, seasonal baseline, date-based split, paired-day bootstrap.
- **[AlaaTarekA/sf-parking-availability](https://github.com/AlaaTarekA/sf-parking-availability)** (Intermediate Status Report): Downtown supply layer, arrival-time target, paid-time features, probability calibration, choice sets and driver replay. Its method was piloted on Melbourne sensor data.

**Area:** the first download for Downtown (Financial District / South Beach) returned only 2,363 payments for 1,661 meters over two months. In the DataSF transactions for Sep-Oct 2026, **IPS meters report no payments**; only MacKay meters do. This repo's own sample shows the same thing: its four silent blocks are all IPS. Downtown is mostly IPS, so the model runs on **Inner Richmond** by default, the neighborhood with the most reporting meters (about 800 MacKay meters), and keeps only MacKay meters. Use `--area` to pick another neighborhood.

## Question

For a block and a driver arriving in 5, 10 or 15 minutes: what is the probability that **at least one meter on the block has no active payment** at arrival? This is a payment-based proxy, not verified free parking (placard holders, non-payers and early departures are invisible).

## Run it

```bash
python src/downtown_supply.py      # blocks, usable meters, 200/400 m choice sets (default --area "Inner Richmond")
python src/download_downtown.py    # session-level payments, Sep 1 - Oct 31, 2026 (needs data.sfgov.org)
python src/train_downtown.py       # models, calibration, baselines, bootstrap, replay (~5-10 min)
python -m unittest discover -s tests
```

Then commit `data/raw/downtown_meters.csv`, `data/raw/downtown_manifest.json`, `data/processed/downtown_blocks.csv`, `results/downtown/` and `results/figures/downtown_*.png`. The raw sessions file and query cache stay local (`.gitignore`).

## Design

| Step | Choice | From |
| --- | --- | --- |
| Area | Inner Richmond by default (`--area`); Downtown has too few reporting meters | Intermediate report, changed after the first download |
| Meters | Active on-street single-space SFMTA MacKay meters, grey or green cap (spaces an ordinary driver may use and that report payments) | Both (this repo's filter + cap-color analysis) |
| Paid status | A meter is paid at time *q* if a payment made at or before *q* covers *q*. Add-time payments extend the end only from when they are made, so later extensions never leak into earlier features | New |
| Target | ≥ 1 meter with no active payment at *t + h*, h = 5, 10, 15 min | Intermediate report |
| Decisions | Every 10 min, 09:00 onward, arrivals by 18:00, Monday–Saturday; block-days with no payments are treated as missing | This repo's window and silent-day rule |
| Features A | Calendar (hour, weekday, Saturday), block size and location, seasonal rate (block × weekday × hour, training days only), paid occupancy 1 day and 7 days earlier | This repo |
| Features B | A + live paid state: occupancy now, unpaid meters now, occupancy 5 and 15 min ago, new sessions in the last 15 min | Intermediate report |
| Features C (RQ2) | B + paid sessions ending within the next 15 minutes (count and share) | Intermediate report |
| Features D (Part 1 → Part 2) | C + Part 1's next-hour forecast for the block (chance of Low and High activity this hour), from Part 1's features and model type (KNN) fitted on the same training dates; out-of-fold for training rows | Both: Part 1's model feeds Part 2 |
| Split | First 60% of dates train, next 20% calibrate, last 20% test | This repo |
| Models | Logistic Regression, Random Forest, Gradient Boosting; isotonic calibration on the calibration dates | Both |
| Baselines | Persistence ("unpaid now stays unpaid"), seasonal rate | Both |
| Primary model | Random Forest, feature set C, calibrated (declared before running) | — |
| Metrics | Brier score (all and busy blocks, 85%+ paid), calibration error, ROC AUC | Intermediate report |
| Uncertainty | Paired-day bootstrap on test dates: primary vs persistence, vs seasonal, C vs B (RQ2), and D vs C (does the next-hour forecast help?) | This repo |
| Replay (RQ3) | 10,000 drivers heading to a busy block, 10 min ahead; choose a block within 400 m minimising walk + (1 − p) × fail cost (3, 6, 10 min) | Intermediate report |

## Outputs

- `results/downtown/supply_summary.json`: Downtown meters by type of use, blocks and choice-set sizes
- `results/downtown/test_metrics.csv`: every model, feature set and horizon
- `results/downtown/summary.json`: split dates, test base rate, Part 1 forecast quality in the area, bootstrap intervals for the four comparisons
- `results/downtown/part1_hourly_forecasts.csv.gz`: Part 1's next-hour forecast for every block-hour used by Part 2
- `results/downtown/replay_10min.csv`: first-try success and average walk per strategy
- `results/figures/downtown_brier_by_horizon.png`, `results/figures/downtown_replay.png`

## Checks

`tests/test_downtown.py` uses small synthetic sessions (never used as results) to check that add-time payments do not leak backwards, that expiring counts use only known end times, that targets come from the arrival time and features from earlier, that splits are in date order with no Sundays, that seasonal rates use training dates only, and the supply filter and choice sets.

## Limitations

- Payment-based proxy, not occupancy; results describe "a meter with no active payment".
- Meter operating hours differ by block and are not verified; 09:00–18:00 is a fixed analysis window.
- Straight-line distances for choice sets; simulated drivers do not compete for the same space.
- The test period is about ten dates; bootstrap intervals are descriptive.
