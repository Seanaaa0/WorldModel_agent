# Hybrid Planning Agent for Partial Observability (PG-WMA)

A modular agent system designed for stable decision-making under partial observability.

This project explores how to combine:

- structured state representation
- memory-based world reconstruction
- high-level planning (LLM)
- low-level execution (rule / predictive)

into a coherent and robust agent architecture.

---

## Core Idea

Instead of letting a single model handle everything:

LLM → decides WHAT to do  
Planner → decides HOW to do it  

This avoids:

- unstable token-level control
- poor reasoning over latent states
- excessive dependence on LLM

---

## System Architecture

```mermaid
graph TD
    A[Observation] --> B[StateEncoder]
    B --> C[WorldMemory]
    C --> D[Planner]
    D --> E[SkillExecutor]
    E --> F[Environment]

    M[Monitor] --> D
```
- StateEncoder → structured latent state  
- WorldMemory → builds global map from partial observations  
- Planner  
  - LLM (slow): phase decision  
  - Rule / Predictive (fast): action execution  
- Monitor → controls stability and replanning  
- SkillExecutor → executes actions  

---

## Components

### StateEncoder

Transforms raw observation into structured state:

- agent position  
- local walls  
- visible objects  
- task state (key / door / goal)  

---

### WorldMemory

Maintains global understanding:

- visited positions  
- known walls / free cells  
- object memory (key / door / goal)  
- BFS path planning  

---

### Monitor

Handles control logic:

- detects oscillation / stuck states  
- triggers replanning  
- handles failure signals  

---

### Planner (Fast–Slow Design)

#### Slow planner (LLM)

Outputs high-level phase:

- find_key  
- go_to_door  
- search_goal  
- go_to_goal  
- recover  

---

#### Fast planner (Rule / Predictive)

Executes actions:

- BFS to known targets  
- frontier exploration  
- local fallback strategies  
- optional predictor rollout  

---

### Predictor (Optional)

Lightweight dynamics model:

- predicts next state  
- improves local decision quality  
- reduces trial-and-error exploration  

---

## Environment

Key-Door-Goal Grid World:

- partial observability (local view only)  
- multi-stage task:
  1. `find key`
  2. `open door`
  3. `reach goal`  

---

## Benchmark Results

All experiments use a 20x20 partially observable key-door-goal grid world unless otherwise specified.

| Experiment | Runs | Success Rate | Avg Steps | Min | Max |
|---|---:|---:|---:|---:|---:|
| Rule V6a | 20 | 100% | 95.35 | 21 | 174 |
| Predictive V8 | 20 | 95% | 94.32 | 25 | 198 |
| Predictive + LLM V8 | 100 | 98% | 100.77 | 20 | 236 |
| Rule V6a, repeat | 20 | 100% | 105.25 | 21 | 253 |

A larger 25x25 Rule V6a run achieved 95% success over 20 runs with 165.63 average steps.

---

## How to Run
```text
python run_agent.py
```

Optional modes (inside code):

- rule  
- predictive  
- llm_slow  
- hybrid (LLM + predictive)  

---

## Project Structure
```text
agent/
  agent_loop.py

encoder/
  state_encoder.py

memory/
  world_memory.py

planner/
  rule_planner.py
  predictive_planner_v8.py
  llm_planner.py

predictor/
  mlp_predictor.py
  jepa_lite_predictor.py

env/
  maze_env.py

skills/
  skill_executor.py
  move_k_steps_skill.py
  escape_loop_skill.py
  base_skill.py
  move_skill.py
  move_until_blocked_skill.py
  scan_skill.py

scripts/
  run_agent.py
  collect_predictor_dataset.py
  train_predictor.py
```

---

## Motivation

This project studies:

- how to stabilize LLM-based agents  
- how to separate planning and execution  
- how to use memory instead of full world models  
- how lightweight predictors improve decision-making  

---

## Key Insight

System design matters more than model size.

Even strong LLMs fail without:

- structured state  
- proper control flow  
- execution constraints  

---

## License

MIT
