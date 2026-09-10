# 1.1 Business Understanding

## Business problem

A bank conducts marketing campaigns to contact existing or prospective customers. A key business question is whether a customer is likely to subscribe to a term deposit (`y = yes`) after a marketing contact.

## Objective

Build a repeatable data pipeline that ingests the Bank Marketing dataset, validates and preprocesses the data, performs exploratory analysis, and produces ML-oriented feature importance using Gradient Boosting.

## Business value

A reliable prediction/analysis pipeline can help a bank understand customer and campaign attributes associated with term-deposit subscription, improve campaign targeting, reduce unnecessary contacts, and support data-driven marketing decisions.

## Target variable

`y` — whether the customer subscribed to a term deposit (`yes`/`no`).

## Pipeline scope

1. Ingest the exact `bank-additional-full.xlsx` dataset used in the ML Lab notebook.
2. Inspect summary statistics, missing values and data types.
3. Remove duplicate records and preprocess numeric/categorical attributes.
4. Normalize numeric values and create meaningful age bins.
5. Calculate correlations and generate univariate/bivariate visualizations.
6. Train a Gradient Boosting Classifier and generate feature importance.
7. Orchestrate the activities with Prefect and run them every two minutes.
