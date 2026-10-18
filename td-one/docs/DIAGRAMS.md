# TD-One Diagrams

Mermaid diagrams for the project. They render on GitHub.

## Pipeline DAG (data → forecast)

```mermaid
flowchart LR
    subgraph INGEST["Ingest (PIT-safe)"]
        FM["Freddie Mac<br/>loan-level"]
        FR["FRED<br/>19 series"]
        NW["News DB<br/>2026, FTS5"]
        SN["Simulator v1/v2<br/>synthetic"]
    end

    subgraph PIT["PIT store (DuckDB)"]
        DUCK["asof joins<br/>no future leakage"]
    end

    subgraph FEAT["Features"]
        COH["Cohort panel<br/>vintage × FICO × rate"]
        MAC["Macro features<br/>z-scored changes"]
        CTX["News context<br/>before=cutoff"]
    end

    subgraph MODELS["Model ladder"]
        M0["M0 / conformal M0"]
        MM["MacroM0"]
        T2["Chronos-2 / TimesFM-2.5"]
    end

    subgraph AGENT["Agentic (LLM: no numbers)"]
        RG["Regime reader"]
        GT["Statistical gate"]
    end

    subgraph OUT["Output"]
        EN["Ensemble<br/>(inverse-CRPS)"]
        FC["Probabilistic forecast<br/>CRPS / Brier / coverage"]
    end

    FM --> DUCK
    FR --> DUCK
    SN --> DUCK
    DUCK --> COH
    FR --> MAC
    NW --> CTX
    COH --> M0
    COH --> MM
    MAC --> MM
    COH --> T2
    CTX --> RG
    RG --> GT
    GT -.->|"intervene / abstain"| EN
    M0 --> EN
    MM --> EN
    T2 --> EN
    EN --> FC
```

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

## Model ladder and verdicts (2026-10-16 snapshot — superseded)

Historical snapshot. Current verdicts: [`experiments/README.md`](../experiments/README.md)
and [`PROPOSED_METHOD.md`](PROPOSED_METHOD.md) (exp47).

| Rung | Model | Verdict |
|------|-------|---------|
| M0 | Beta-Binomial baseline (tuned window, conformalized) | Default baseline. Strong on synthetic, loses to TSFMs on real data |
| M1 | Chronos-2 / TimesFM-2.5 (real weights) | Lose on synthetic; **win ~20% on real M4 data** |
| M2 | ReAct agent (numerical forecaster) | **Killed** — honest negative with real backend |
| M3 | Hybrid (LLM deltas + gate) | Gate helps; default-adjust killed |
| M4 | Regime reader (LLM sensor → statistical gate) | Qualified positive (3.59 vs 4.82, real Groq) |
| M5 | Ensemble router (REATS-style) | **Killed** — clean negative vs inverse-CRPS |

## The agentic trilogy → proposed method (exp42–47)

```mermaid
flowchart TB
    subgraph T42["exp42 · FFORMA meta-learner"]
        F42["Learn combination weights<br/>from meta-features"]
        V42["❌ honest negative<br/>p=0.156, loses to equal weights"]
    end
    subgraph T43["exp43 · Constrained regime router"]
        F43["Hand rules → LogisticReg<br/>→ P(regime | features)<br/>Σ P(r) × expert_r"]
        V43["✅ partial positive<br/>M4 0.1282 vs 0.1295, p=0.018"]
    end
    subgraph T44["exp44 · News as selector"]
        F44["News stress → shift P(regime)<br/>toward stressed/noisy"]
        V44["⚖️ mechanism proven synthetic<br/>(p=0.0068); real blocked<br/>by news sparsity"]
    end
    subgraph T45["exp45 · Learned regimes"]
        F45["Unsupervised clustering<br/>instead of hand rules"]
        V45["❌ honest negative<br/>hand rules win, p=0.00046"]
    end

    F42 --> V42
    F43 --> V43
    F44 --> V44
    F45 --> V45

    V43 --> P47["exp47 · Proposed method<br/>router + gates + conformal<br/>✅ 0.1451 vs 0.1507, p=6.5e-09"]
    V44 -.->|"gated, off by default"| P47

    style V42 fill:#f9d5d5
    style V43 fill:#d5f9d5
    style V45 fill:#f9d5d5
    style P47 fill:#d5f9d5
```

Full architecture: [PROPOSED_METHOD.md](PROPOSED_METHOD.md).

## Tool-tracking loop (agent infrastructure)

```mermaid
flowchart LR
    TOOLS["10 agent tools<br/>src/td_one/agent/tools/"]
    TRACK["@tracked() decorator<br/>src/td_one/agent/tool_tracker.py<br/>timing · hashed inputs<br/>outputs · success/fail"]
    DATA["TOOL_USAGE_DATA.md<br/>runtime call log"]
    AUDIT["TOOL_AUDIT.md<br/>frequency · patterns · failures"]
    OPT["TOOL_OPTIMIZATION.md<br/>add / drop / fix<br/>before final report"]

    TOOLS --> TRACK --> DATA --> AUDIT --> OPT
    OPT -.->|"implement"| TOOLS
```

## News cache enrichment (live news track)

```mermaid
flowchart LR
    GDELT["GDELT fetcher<br/>scripts/fetch_news_gdelt.py"]
    PERP["Perplexity fetcher<br/>(credits: avoid)"]
    CACHE[("news cache<br/>data/news/<br/>FTS5 SQLite")]
    ENRICH["enrich on every hit"]
    EXP["experiments<br/>news context<br/>before=cutoff"]

    GDELT --> ENRICH --> CACHE
    PERP --> ENRICH
    CACHE --> EXP

    style PERP fill:#fff3d5
```

## Experiment timeline, continued (Oct 17–18)

```mermaid
timeline
    title TD-One experiments (Oct 17–18, 2026)
    2026-10-17 : R2 PIT-safe reruns (macro track closed)
               : exp16–18b Freddie SMM real-data benchmark
               : exp19 news-SMM frozen
               : exp20–25 event-augmented track
               : exp40–41 calibration + meta-learning
    2026-10-18 : exp42 FFORMA (negative)
               : exp43 regime router (partial positive)
               : exp44 news selector (partial positive)
               : exp45 learned regimes (negative)
               : exp46 dense news (definitive negative)
               : exp47 proposed method (positive)
               : exp48 reasoning sweep (running)
```
