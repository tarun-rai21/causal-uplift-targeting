-- db/schema.sql

-- Causal Uplift Targeting: relational schema
-- SQLite-compatible, Postgres-portable (raw SQL kept explicit — Core, not ORM —
-- both for a one-line Postgres swap later and as a visible SQL-fluency artifact).

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    signup_date DATE NOT NULL,
    segment TEXT,                      -- ground-truth label, simulation only; unused on real data
    historical_spend REAL,
    region TEXT NOT NULL,
    device_type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pilots (
    pilot_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    start_date DATE NOT NULL,
    budget_cap REAL,                   -- max spend available for a real rollout
    ground_truth_json TEXT             -- simulated pilots only: frozen GroundTruth; NULL on real data
);

CREATE TABLE IF NOT EXISTS treatment_assignment (
    user_id TEXT NOT NULL REFERENCES users(user_id),
    pilot_id TEXT NOT NULL REFERENCES pilots(pilot_id),
    treated BOOLEAN NOT NULL,
    propensity_e REAL NOT NULL,        -- known randomization probability, e.g. 0.5
    assigned_at TIMESTAMP NOT NULL,
    PRIMARY KEY (user_id, pilot_id)
);

CREATE TABLE IF NOT EXISTS outcomes (
    user_id TEXT NOT NULL REFERENCES users(user_id),
    pilot_id TEXT NOT NULL REFERENCES pilots(pilot_id),
    converted BOOLEAN NOT NULL,
    revenue REAL NOT NULL DEFAULT 0.0,
    observed_at TIMESTAMP NOT NULL,
    PRIMARY KEY (user_id, pilot_id)
);

-- Phase 1.5 addition (not in original spec): persists which split each user/pilot
-- row belongs to, so every downstream phase (baseline, T-learner, X-learner,
-- evaluation, validation) evaluates against a consistent, reproducible held-out set
-- instead of silently re-deriving or leaking across a mix of ad hoc splits.
CREATE TABLE IF NOT EXISTS data_splits (
    user_id TEXT NOT NULL REFERENCES users(user_id),
    pilot_id TEXT NOT NULL REFERENCES pilots(pilot_id),
    split TEXT NOT NULL,               -- 'train' | 'test'
    split_seed INTEGER NOT NULL,
    PRIMARY KEY (user_id, pilot_id)
);

CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id TEXT PRIMARY KEY,
    pilot_id TEXT NOT NULL REFERENCES pilots(pilot_id),
    model_type TEXT NOT NULL,          -- 'baseline' | 't_learner' | 'x_learner'
    base_learner TEXT NOT NULL,        -- 'gradient_boosting' | 'logistic_regression'
    trained_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS uplift_predictions (
    model_run_id TEXT NOT NULL REFERENCES model_runs(model_run_id),
    user_id TEXT NOT NULL REFERENCES users(user_id),
    predicted_uplift REAL NOT NULL,
    rank_pct REAL NOT NULL,            -- percentile rank within this model_run, precomputed for Qini
    PRIMARY KEY (model_run_id, user_id)
);

-- Phase 5 addition: one row per evaluated model_run. Scalar scores and their
-- paired-bootstrap intervals; the curve itself lives in qini_curve_points.
-- Intervals reflect evaluation-sample noise only (fitted model held fixed).
CREATE TABLE IF NOT EXISTS model_evaluations (
    model_run_id TEXT PRIMARY KEY REFERENCES model_runs(model_run_id),
    n_users INTEGER NOT NULL,          -- test-set size the scores were computed on
    qini_coefficient REAL NOT NULL,    -- mean of Q(t) minus the random line, t = 0..N
    qini_ci_lower REAL NOT NULL,
    qini_ci_upper REAL NOT NULL,
    auuc REAL NOT NULL,                -- mean of the cumulative-gain curve G(t)
    auuc_ci_lower REAL NOT NULL,
    auuc_ci_upper REAL NOT NULL,
    n_bootstrap INTEGER NOT NULL,
    ci_level REAL NOT NULL,
    bootstrap_seed INTEGER NOT NULL,
    evaluated_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS qini_curve_points (
    model_run_id TEXT NOT NULL REFERENCES model_runs(model_run_id),
    pct_targeted REAL NOT NULL,
    cumulative_incremental_gain REAL NOT NULL,
    ci_lower REAL,
    ci_upper REAL,
    PRIMARY KEY (model_run_id, pct_targeted)
);

-- Phase 7: shared per-user economics the policy layer prices targeting with
-- (core/economics.py), for the pilot's test users. Replaced on each pipeline run.
CREATE TABLE IF NOT EXISTS user_economics (
    pilot_id TEXT NOT NULL REFERENCES pilots(pilot_id),
    user_id TEXT NOT NULL REFERENCES users(user_id),
    p_convert_if_treated REAL NOT NULL,     -- P(convert | treated, x)
    revenue_per_conversion REAL NOT NULL,   -- E[revenue | converted, x]
    PRIMARY KEY (pilot_id, user_id)
);

-- Phase 7: one row per budget point of a policy curve (core/policy.py).
-- Models choose WHO is targeted; the randomized evaluation-set outcomes value
-- it (ab_* columns, unbiased). model_* are the model's own valuations over the
-- users it picked -- optimistic by construction, kept for contrast; NULL for
-- the naive baseline, which has no uplift estimate.
CREATE TABLE IF NOT EXISTS policy_simulations (
    policy_sim_id TEXT PRIMARY KEY,
    policy_curve_id TEXT NOT NULL,     -- groups the points of one curve
    model_run_id TEXT NOT NULL REFERENCES model_runs(model_run_id),
    cashback_rate REAL NOT NULL,       -- share of revenue paid on each treated conversion
    min_roi REAL,                      -- ROI floor applied, NULL if none
    budget_pct REAL NOT NULL,          -- budget / estimated cost of treating everyone
    budget REAL NOT NULL,
    n_targeted INTEGER NOT NULL,
    expected_spend REAL NOT NULL,      -- planning estimate the selection stopped on
    model_incremental_revenue REAL,
    model_net_revenue REAL,
    model_marginal_roi REAL,           -- d(model revenue)/d(spend) since the previous point
    ab_incremental_revenue REAL,       -- NULL when the targeted set lacks a treated or control user
    ab_spend REAL,
    ab_net_revenue REAL,
    ab_net_ci_lower REAL,
    ab_net_ci_upper REAL,
    created_at TIMESTAMP NOT NULL
);