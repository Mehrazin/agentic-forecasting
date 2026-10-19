# Freddie Mac SMM

63-month SMM benchmark. Fair ablation (exp18b): calibrated TimesFM beats the
best classical (ETS+residuals) by 12.2% (0.000497 vs 0.000566, p=0.076 n.s.) —
directional edge, not significant. exp33 (agent features +1.21%, n.s.),
exp34 (real BoC events hurt: −13.04%, priced in). News augmentation (exp19)
frozen — news doesn't move SMM.

Files: [`RESULTS.md`](RESULTS.md), `exp33_agent_features.py`, `exp34_real_events.py` (+ JSONs).
Up: [`benchmarks/`](../README.md).
