---
title: Uganda Coffee Price Predictor
emoji: ☕
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: "1.53.0"
app_file: app.py
pinned: false
---

# Uganda Coffee Price Predictor

Predicting Ugandan coffee prices by grade — Bugisu AA/A/B (Arabica) and
Screen 18/15/12 (Robusta) — from the forces that move them: world
Arabica/Robusta prices, the USD/UGX rate, Brent crude, and US Fed policy.

Shows live prices across all 12 UCDA-tracked grades (export FOB and farmgate),
price history, driver correlations, and a baseline regression evaluated with
walk-forward cross-validation.

⚠️ The 30-month grade history is approximate (static fallback data from the
source page, not a live UCDA feed), and there are effectively 2 independent
series rather than 6. See the in-app data-quality note and the main repo
README before drawing conclusions.

Source and methodology:
https://github.com/<your-username>/uganda-coffee-price-predictor
