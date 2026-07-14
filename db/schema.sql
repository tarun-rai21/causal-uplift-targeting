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
    budget_cap REAL                    -- max spend available for a real rollout
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

CREATE TABLE IF NOT EXISTS qini_curve_points (
    model_run_id TEXT NOT NULL REFERENCES model_runs(model_run_id),
    pct_targeted REAL NOT NULL,
    cumulative_incremental_gain REAL NOT NULL,
    ci_lower REAL,
    ci_upper REAL,
    PRIMARY KEY (model_run_id, pct_targeted)
);

CREATE TABLE IF NOT EXISTS policy_simulations (
    policy_sim_id TEXT PRIMARY KEY,
    model_run_id TEXT NOT NULL REFERENCES model_runs(model_run_id),
    budget_pct REAL NOT NULL,
    n_targeted INTEGER NOT NULL,
    expected_incremental_revenue REAL NOT NULL,
    expected_incremental_revenue_ci_lower REAL,
    expected_incremental_revenue_ci_upper REAL,
    marginal_roi REAL NOT NULL         -- d(incremental revenue)/d(budget) at this point
);