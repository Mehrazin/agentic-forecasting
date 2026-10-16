# TD-One Diagrams

Mermaid diagrams for the project. They render on GitHub.

## System architecture (from REDESIGN.md)

```mermaid
flowchart TB
    subgraph DATA["Data layer"]
        FRED["FRED macro series<br/>(19 series)"]
        SIM["Simulator v1 / v2<br/>(synthetic mortgages)"]
        M4["M4 Monthly<br/>(real benchmark)"]
        FM["Freddie Mac<br/>(real mortgages, pending)"]
    end

    subgraph CLASSICAL["Classical models"]
        M0["M0 Beta-Binomial<br/>(tuned window)"]
        CONF["Conformalized M0<br/>(default baseline)"]
        MACRO["MacroM0<br/>(FRED OLS overlay)"]
        HIER["Hierarchical<br/>(thin segments)"]
    end

    subgraph TSFM["Foundation models"]
        C2["Chronos-2<br/>(real weights)"]
        TFM["TimesFM-2.5<br/>(real weights)"]
        C2M["Chronos-2 macro<br/>(residual conditioning)"]
    end

    subgraph AGENTIC["Agentic layer — LLM never outputs numbers"]
        ROUTER["M5 Ensemble router<br/>(weights + rationale)"]
        REGIME["M4 Regime reader<br/>(class + confidence)"]
        GATE["Statistical gate<br/>(z-test, abstention)"]
    end

    FRED --> MACRO
    FRED --> C2M
    SIM --> M0
    SIM --> C2
    SIM --> TFM
    M4 --> C2
    M4 --> TFM

    M0 --> CONF
    MACRO --> ENS["Ensemble<br/>(inverse-CRPS)"]
    CONF --> ENS
    C2 --> ENS
    TFM --> ENS

    ROUTER -.->|"weights (killed: lost to inv-CRPS)"| ENS
    REGIME -->|"regime + confidence"| GATE
    GATE -->|"intervene / abstain"| ENS

    ENS --> OUT["Probabilistic forecast<br/>(CRPS / Brier / coverage)"]

    style ROUTER fill:#f9d5d5
    style CONF fill:#d5f9d5
    style REGIME fill:#d5f9d5
```

## Experiment timeline

```mermaid
timeline
    title TD-One experiments (Oct 15–16, 2026)
    2026-10-15 : exp1 pipeline validation (mock)
               : exp2 v2 M0 vs real TSFMs (5 seeds)
               : exp3 M2 ReAct (mock + real Groq)
               : exp4 M3 hybrid (mock + real Groq)
               : Scale-up 10k×66
    2026-10-16 : exp5 better baselines (conformal wins)
               : exp6 M3 over-adjustment fixes (gate wins)
               : exp8 macro DGP (FRED overlay, 32% gain)
               : exp9 simulator v2 (window ranking reverses)
               : exp10 classical track (conformal default)
               : exp11 M4 regime reader (qualified positive)
               : exp12 M5 ensemble router (clean negative)
               : exp13 TSFM+FRED rematch (draw)
               : exp14 M4 real data (synthetic story breaks)
```

## Key result: synthetic vs real

```mermaid
flowchart LR
    subgraph SYN["Synthetic DGPs (exp2, exp5, multi-seed)"]
        S1["M0 (tuned) beats<br/>Chronos-2, TimesFM-2.5<br/>no significance, 5 seeds"]
    end
    subgraph REAL["Real data (exp14, M4 Monthly, n=2000)"]
        R1["Chronos-2 350.85<br/>TimesFM-2.5 356.35<br/>vs naive 445.40<br/>TSFMs win by ~20%"]
    end
    SYN -.->|"DGP too easy for M0<br/>(survey §5)"| REAL

    style S1 fill:#fff3d5
    style R1 fill:#d5f9d5
```

## Model ladder and verdicts

| Rung | Model | Verdict |
|------|-------|---------|
| M0 | Beta-Binomial baseline (tuned window, conformalized) | Default baseline. Strong on synthetic, loses to TSFMs on real data |
| M1 | Chronos-2 / TimesFM-2.5 (real weights) | Lose on synthetic; **win ~20% on real M4 data** |
| M2 | ReAct agent (numerical forecaster) | **Killed** — honest negative with real backend |
| M3 | Hybrid (LLM deltas + gate) | Gate helps; default-adjust killed |
| M4 | Regime reader (LLM sensor → statistical gate) | Qualified positive (3.59 vs 4.82, real Groq) |
| M5 | Ensemble router (REATS-style) | **Killed** — clean negative vs inverse-CRPS |
