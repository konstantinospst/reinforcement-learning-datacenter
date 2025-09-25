# reinforcement-learning-datacenter
# Data Center Energy Agent — RL Optimization

> Course project exploring **Tabular Q-Learning** and **Deep Q-Networks (DQN)** for optimizing a data center’s buy/sell/storage decisions under hourly electricity prices.

**Authors:** Athanasios Katranis (2803183), Konstantinos Pasatas (2803568), Mert Unveren (2709757)  
**Institution:** Vrije Universiteit Amsterdam

---

## Overview

We formulate day-ahead operations as an RL problem with:
- **State**: `[storage_level (MWh), price (€/MWh), hour (1–24), day]`
- **Action**: continuous in `[-1, 1]` mapped to buy/hold/sell at up to **10 MW** (±10 MWh) per hour
- **Daily demand**: **120 MWh** target with carry-over cap of **+50 MWh**
- **Environment rules**: forced buys if the remaining hours cannot meet demand; disallow sells that would make demand infeasible.

We compare:
- **Tabular Q-Learning** with discretized storage/price and explicit hour dimension.
- **DQN** with PER, n-step returns, target network, and soft updates.

**Result (summary):** On our dataset/splits, the **tabular agent** achieved the best performance; DQN shows promising behavior but needs further tuning and reward shaping.  

---

## Repository Structure

