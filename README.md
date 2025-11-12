# Double Inverted Pendulum RL (dip_rl)

A reinforcement learning project for training agents to balance a double inverted pendulum using function-approximated Q-Learning (LFA) and policy-gradient methods (PPO from Stable-Baselines3). Built with PyBullet, Gymnasium, and ROS2-compatible simulation structure.

---

## 🧠 Features

- ✅ **Custom Gymnasium Environment** with PyBullet physics.
- ✅ **Proximal Policy Optimization (PPO)** via Stable-Baselines3.
- ✅ **Q-Learning with Linear Function Approximation (LFA)**.
- ✅ **Training Mode Toggle**: switch between deterministic dev mode and randomized robust mode.
- ✅ **Evaluation Utilities**: success rate, reward curves, and TensorBoard metrics.
- ✅ **Modular Callbacks**: checkpointing, early stopping, custom TensorBoard logging.

---

## 🚀 Getting Started
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train PPO Agent
```bash
python train.py --mode dev       # or --mode robust
```

### 3. Run LFA-Based Q-Learning
```bash
python q_learning_lfa.py --mode dev       # or --mode robust
```
---


## 📁 Repository Structure

```bash
dip_rl/
├── dip_env/                         # Custom Gymnasium-compatible environment
│   └── dip_env.py
├── models/                          # URDF model of double pendulum
│   └── double_pendulum.urdf
├── logs/                            # TensorBoard logs, checkpoints, eval results
├── train.py                         # PPO training script (with toggleable mode)
├── q_learning_lfa.py                # Q-Learning (LFA) implementation
└── README.md
```
---

## 📌 References
Gustafsson et al., Control of Inverted Double Pendulum using Reinforcement Learning

---

## 👨‍🔬 Maintainer Info
- Maintainer: Samuel Chien

- Lab: Mechatronics and Controls Laboratory, UCLA

- Email: samuelbruin0618@g.ucla.edu
