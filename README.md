#  WorldModel_Agent V4 
  — LLM + Predictive Hybrid Planner

##  Overview

Maze Agent V4 is a hybrid decision-making system that integrates:

* **Rule-based planning (fast, stable)**
* **Predictive model (local rollout)**
* **LLM-based slow planner (global reasoning attempt)**

The goal of this version is to evaluate whether a language model can improve decision quality in a partially observable navigation task.

---

##  Architecture

```
Observation (z_t)
    ↓
Memory Update
    ↓
Planner Selection
    ├── Fast Planner (Rule + Predictor)
    └── Slow Planner (LLM)
    ↓
Action Execution
    ↓
Environment Feedback
```

### Components

* **Agent Loop**

  * Maintains memory
  * Handles decision flow
  * Supports debug logging

* **Fast Planner (Predictive Rule)**

  * Heuristic scoring
  * One-step prediction rollout
  * Loop avoidance

* **Slow Planner (LLM-based)**

  * Prompt-based decision making
  * Uses memory + local context
  * Designed for higher-level reasoning

* **Predictor**

  * Estimates next state
  * Provides local transition hints

---

##  Experiment Setup (V4)

* Environment: `15x15 Grid`
* Seeds: `0–19 (fixed)`
* Max steps: fixed per episode
* Comparison:

| Planner                | Description           |
| ---------------------- | --------------------- |
| fast_predictive_legacy | heuristic + predictor |
| llm_slow               | LLM-assisted planning |

---

##  Results

### LLM Slow Planner

* Success Rate: **0.90**
* Avg Steps (success): **86.78**

### Fast Predictive (Legacy)

* Success Rate: **1.00**
* Avg Steps (success): **97.35**

---

##  Key Findings

* The **predictive rule-based planner remains more stable** (100% success).
* The **LLM planner achieves lower average steps** on successful runs.
* However, LLM fails on harder long-horizon cases.

### Conclusion

> LLM does not outperform heuristic + predictive planning in low-level navigation tasks.

Instead, it shows potential in:

* reducing path length
* making more direct decisions (when correct)

---

##  Limitations

* LLM is used in a **single-step decision mode**
* No multi-step planning or rollout
* No uncertainty modeling
* Environment is still **pure navigation (no task abstraction)**

---

##  Next Direction (V5)

The next version will shift from:

```
Navigation Problem → Task Execution Problem
```

Planned upgrades:

* Skill-based action space
* Task-oriented environment (e.g., key-door interactions)
* Multi-step planning (rollout)
* Improved world model (predictor V2)

---

##  How to Run

```bash
python -m run.run_agent
```

To analyze results:

```bash
python visual/analyze_results.py
```

---

##  Project Structure (simplified)

```
agent/
encoder/
memory/
skill/
scripts/
monitor/
run/
planner/
predictor/
env/
run/
visual/
```

---

##  Version History

* V1: Basic rule-based agent
* V2: Memory + improved heuristics
* V3: Predictor integration
* V4: LLM + predictive hybrid comparison

---

##  Author

Sean Hsu
AI Developer — Reinforcement Learning × LLM Systems

---
"# WorldModel_agent" 
