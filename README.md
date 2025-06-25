# Reinforcement Learning Algorithms in GridWorld

This repository contains the work from my Reinforcement Learning internship at the **Laboratoire J.A. Dieudonné** (Université Côte d'Azur), conducted under the supervision of M. François Delarue, recipient of the 2022 Joseph L. Doob Prize from the American Mathematical Society.

## Project Overview

Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make optimal decisions through trial and error, interacting with an environment to maximize cumulative rewards.

This project implements and compares several fundamental RL algorithms, using a stochastic GridWorld environment as a testbed. The goal is to find the optimal strategy (policy) under different assumptions about the environment.

### Key Features
- **Implementation of Core RL Algorithms:** From classic planning to model-free learning.
- **Stochastic Environments:** Agents must handle uncertainty, as actions are not always deterministic.
- **Complex Navigation:** Environments include obstacles to create non-trivial pathfinding challenges.
- **Interactive Pygame Visualization:** A game built with Pygame to visualize the learned policies of the agents in action or to play the game manually.

---

## Algorithms Implemented

The project explores two primary scenarios:

### 1. Known Model (Planning)
*When the agent has a perfect map of the environment.*

- **Value Iteration:** An algorithm that computes the optimal value function and derives the best possible policy.
- **Policy Evaluation:** Calculates the expected utility of following a specific, given policy.

### 2. Unknown Model (Reinforcement Learning)
*When the agent must learn by exploring the environment.*

- **Passive Agents (Policy Evaluation):**
    - **Adaptive Dynamic Programming (ADP):** A model-based agent that learns a model of the environment and uses it to plan.
    - **Temporal Difference (TD) Learning:** A model-free agent that learns directly from experienced transitions.
- **Active Agent (Control):**
    - **Q-Learning:** A model-free agent that learns the optimal policy directly without needing a model.

---

## Interactive GridWorld Game

To better understand and visualize the results, I developped a game using **Pygame**.

The game allows you to:
- **Watch the agents** navigate the grid according to their learned policies.
- **Play the game yourself** to understand the challenges the agents face.
- **Check it out** in the `/game` directory!
