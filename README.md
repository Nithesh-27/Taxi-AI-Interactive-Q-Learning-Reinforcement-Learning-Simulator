# 🚕 Taxi AI – Q-Learning Reinforcement Learning Simulator

An interactive Reinforcement Learning simulator that demonstrates how an AI taxi agent learns optimal navigation strategies using the **Q-Learning algorithm**.

The project provides a **visual environment**, **training metrics**, and **performance analysis** to help understand how Reinforcement Learning agents learn through exploration and exploitation.

---

## 📌 Features

- Interactive **Taxi Grid Environment**
- **Q-Learning based reinforcement learning agent**
- Real-time **training visualization**
- **Performance tracking chart**
- **Exploration vs Exploitation visualization**
- **Random agent comparison mode**
- Adjustable **training speed**
- Dark / Light UI modes
- Downloadable **training results**

---

## 🧠 Reinforcement Learning Concept

This project implements **Q-Learning**, a model-free reinforcement learning algorithm.

The agent learns by interacting with the environment and updating a **Q-Table**.

### Q-Learning Update Rule

```
Q(s,a) ← Q(s,a) + α [ r + γ max Q(s',a') − Q(s,a) ]
```

Where:

| Symbol | Meaning |
|------|------|
| s | current state |
| a | action |
| r | reward |
| s' | next state |
| α | learning rate |
| γ | discount factor |

---

## 🧭 Environment

The simulator uses a **5x5 grid environment** where the taxi must:

1. Navigate to the passenger
2. Pick up the passenger
3. Deliver them to the destination

### Possible Actions

| Action | Description |
|------|------|
| 0 | Move North |
| 1 | Move South |
| 2 | Move East |
| 3 | Move West |
| 4 | Pickup Passenger |
| 5 | Dropoff Passenger |

### Reward System

| Event | Reward |
|------|------|
| Each step | -1 |
| Wrong pickup/dropoff | -10 |
| Successful dropoff | +20 |

---

## 📊 Visualization

The simulator shows:

- Agent movement in real time
- Heatmap of learned Q-values
- Best action arrows
- Episode statistics
- Average reward history

---

## 🧪 Headless Verification

The project includes a **headless simulation script** to verify the learning logic.

Run:

```bash
node verify_logic.js
```

Example Output:

```
Episode 100: Avg Reward = -198
Episode 200: Avg Reward = -57
Episode 300: Avg Reward = -28
Episode 400: Avg Reward = -9
Episode 500: Avg Reward = -1
```

This indicates that the agent is **progressively improving its policy**.

---

## 🚀 How to Run

### Option 1 — Direct Browser

1. Download the project
2. Open:

```
index.html
```

The simulator will launch automatically.

---

### Option 2 — Local Server (Recommended)

Using VS Code Live Server:

```
Right Click index.html
→ Open with Live Server
```

---

## 🎮 Controls

| Control | Function |
|------|------|
Start Training | Begins agent training |
Pause | Pauses simulation |
Reset | Resets the environment |
Train/Test Mode | Switch between training and evaluation |
Speed Slider | Adjust training speed |
Comparison Mode | Compare with random agent |

---

## 📂 Project Files

| File | Description |
|------|------|
index.html | UI layout and dashboard |
styles.css | Visual styling and theme |
script.js | RL logic, environment, visualization |
verify_logic.js | Headless RL verification |
simulation_output.txt | Sample training output |

---

## 📈 Learning Parameters

| Parameter | Value |
|------|------|
Learning Rate (α) | 0.2 |
Discount Factor (γ) | 0.9 |
Initial Epsilon | 1.0 |
Epsilon Decay | 0.9995 |
Minimum Epsilon | 0.01 |

---

## 🛠 Technologies Used

- HTML5
- CSS3
- JavaScript
- Canvas API
- Reinforcement Learning
- Q-Learning Algorithm

---

## 🎯 Educational Purpose

This project is designed to help understand:

- Reinforcement Learning fundamentals
- Exploration vs Exploitation
- Q-table learning
- Agent environment interaction
- Policy convergence

---

## 📌 Future Improvements

- Deep Q-Network (DQN)
- Multiple taxi agents
- Larger grid environments
- Path optimization comparison
- Performance benchmarking

---

## 👨‍💻 Author

**Nithesh K**

Machine Learning Enthusiast  
AI / Data Science

---

## ⭐ If you like this project

Give it a **star ⭐ on GitHub** to support the project.
