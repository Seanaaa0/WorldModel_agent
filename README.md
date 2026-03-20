# WorldModel Agent V6
Fast–Slow Maze Agent with LLM Phase Planning + Predictive Execution

---

## Overview

Maze Agent V6 is a **fast–slow hybrid agent system** designed to study decision-making under partial observability.

Unlike previous versions that mixed decision layers, V6 introduces a **clean separation between:**

- **Slow planner (LLM)** → phase-level reasoning
- **Fast planner (Predictive)** → local action execution

This version serves as a **final toy testbed** validating that architecture.

---

## Core Idea

Instead of letting LLM choose every action, we enforce:

```text
LLM → "what to do"
Fast planner → "how to do it"
```

This avoids:

- unstable token-level control
- poor handling of latent state
- excessive LLM dependency

---

## Architecture

```text
Environment
    ↓
State Encoder (z_t)
    ↓
Memory Update
    ↓
Monitor (event detection)
    ↓
Slow Planner (LLM, phase-level, event-driven)
    ↓
Phase Controller
    ↓
Fast Planner (PredictivePlannerV8)
    ↓
Skill Execution
    ↓
Environment Feedback
```

---

## Components

### Agent Loop
- Orchestrates the full pipeline
- Handles monitor-triggered replanning
- Maintains phase state

### Monitor
- Detects key events:
  - picked key
  - opened door
  - stuck / oscillation
- Triggers **slow replanning only when needed**

### Slow Planner (LLM)
- Outputs **phase-level decisions**
- Examples:
  - `find_key`
  - `go_to_door`
  - `search_goal`
  - `go_to_goal`
  - `recover`
- Not used for primitive actions

### Fast Planner (PredictivePlannerV8)
- Local action selection
- Uses:
  - memory
  - frontier exploration
  - predictor-guided evaluation
- Executes behavior conditioned on the current phase

### Predictor
- One-step state prediction
- Provides local rollout signals
- Used as a **decision aid**, not a full planning engine

### Recover System
- First-class control mode
- Loop-aware escape mechanism
- Handles oscillation / stuck cases

---

## Experiment Setup (V6)

- Environment: `20x20 Grid`
- Wall probability: `0.12`
- View radius: `3`
- Max steps: `300`
- Mode: `predictive_v8_llm_phase`

---

## Results

### 20 Seeds
- Success: **20 / 20**
- Success rate: **1.00**

### 100 Seeds
- Success: **98 / 100**
- Success rate: **0.98**
- Avg steps (success): **~100.8**

---

## Key Findings

### 1. Fast–Slow separation works
- LLM is effective at **phase planning**
- Fast planner is more reliable for **execution**

### 2. Event-driven replanning is critical
- Continuous LLM usage is unnecessary
- Monitor-triggered replanning is sufficient and efficient

### 3. Recover must be explicit
- Recovery is not optional
- Loop-aware control significantly improves robustness

### 4. Predictor is useful but limited
- Improves local decisions
- Not sufficient as a full planning engine

---

## Failure Analysis

Remaining failures (2/100):

- Rare local navigation edge cases
- Exploration coverage issues

These are **tail cases**, not architecture failures.

---

## Why This Version Stops Here

This project has achieved its goal as a **toy fast–slow architecture validation**.

Key questions answered:

- Can LLM operate at phase level instead of action level? → Yes
- Can the fast planner handle execution reliably? → Yes
- Can monitor-driven replanning work? → Yes
- Is recovery necessary? → Yes

Further improvements would mostly involve **overfitting rare seeds**, which is not the goal.

---

## Next Direction

Shift from:

```text
Maze Navigation → Real Task Environment
```

Future work:

- tool / task-based environment
- richer state semantics
- multi-step planning / rollout
- integration with learned world models (e.g., JEPA)

---

## How to Run

```bash
python -m run.run_agent
```

Analyze results:

```bash
python visual/analyze_results.py
```

---

## Project Structure (simplified)

```text
agent/
encoder/
memory/
monitor/
planner/
predictor/
skill/
env/
run/
visual/
```

---

## Version History

- V1: Rule-based agent
- V2: Memory + heuristics
- V3: Predictor integration
- V4: LLM vs Predictive comparison
- V5: Phase-based planning (initial)
- V6: **Fast–Slow architecture (final toy version)**

---

## Author

Sean Hsu  
