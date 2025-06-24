# 🧠 Intelligent Maze Navigation with PyBullet

This project is part of the **MAI5100** course within the **MSc Artificial Intelligence Program** at the **University of Guyana, SA**.

It uses the [PyBullet](https://pybullet.org/) Python library to simulate an intelligent bot that performs **dynamic path planning** to navigate a randomly generated maze. The bot uses **A\*** and **CSP-based strategies** to reach multiple goals while avoiding randomly moving neutral agents.

---

## 🚀 Features

* **Environment Generation**

  * Randomly generated maze with customizable size
  * Multiple goals with different deadlines
  * Obstacles (walls) and enemy agents

* **Path Planning Algorithms**

  * Basic A\* Search
  * Greedy multi-goal A\*
  * A\* with constraints (deadlines)

* **Enemy Agents**

  * Move randomly each timestep
  * Detection via raycasting

* **Visualization**

  * 3D rendering of maze, agents, and bot using PyBullet
  * Realtime text overlays and path updates

---

## 📁 Project Structure

| File               | Description                                                                        |
| ------------------ | ---------------------------------------------------------------------------------- |
| `main.py`          | Main simulation script: initializes maze, performs pathfinding, and runs PyBullet. |
| `mazeGenerator.py` | Handles maze generation, obstacle and agent placement, and goal assignment.        |
| `agent.py`         | Contains search algorithms: A\*, Greedy A\*, and CSP-enhanced planning.            |
| `graphics.py`      | Handles PyBullet rendering, bot movement, raycasting, and debugging visuals.       |

---

## 🛆 Requirements

* Python 3.7+
* `pybullet`
* `numpy`
* `random` (standard lib)

To install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ How It Works

### 🔧 Maze Generation

A grid-based environment is created with:

* Obstacles (black boxes)
* Enemy agents (blue cubes)
* Goals (green tiles)

### 🌟 Goal Assignment

Each goal is randomly assigned:

* A **deadline** (must reach within a fixed time), or
* An **"any"** deadline (flexible timing)

### 🧠 Pathfinding

The bot uses:

* **Greedy A\*** with **CSP-DFS** for goal sequencing
* **Replanning** whenever enemies are detected via raycast
* **Raycasting** for reflex-based threat detection
* **Smooth animation** via interpolated movement

### 🤖 Enemy Agents

Each agent moves randomly per timestep, forcing the bot to dynamically adapt.

---

## 🎮 Controls & Interaction

* Run the simulation:

```bash
python main.py
```

* Press **Enter** in the terminal when prompted to end the simulation.
* Alerts (e.g. no valid path) are printed in the terminal.

---

## 📊 Algorithms Used

* **A\***: Manhattan distance heuristic.
* **Greedy A\***: Picks the closest goal at each step.
* **CSP-based Planning**:

  * Schedules paths respecting deadlines using DFS
  * Avoids zones with dynamic agents

---

## 🛠️ Customization

You can tune parameters in `main.py`:

```python
gridSize = 12       # Size of the maze
obst_prob = 0.1     # Obstacle density
num_agents = 7      # Number of enemy agents
num_goals = 3       # Number of goal tiles
```

---

## ⚠️ Notes

* If an agent blocks the current path, the bot will **recompute a new path**.
* If **no path is found**, a warning is printed and you may rerun `main.py`.
* The simulation is **non-deterministic**: not all maze setups are solvable.

---

## 🎥 Demo Video

Watch a demonstration:
👉 [https://youtu.be/-KhcPkbhVoQ](https://youtu.be/-KhcPkbhVoQ)

---

## 🖼️ Screenshots

> *(Add image files to your repository and update the image paths below)*

**Simulation Overview**

```
[Bot 🟡]  → Navigating  
[Walls ⬛] → Obstacles  
[Goals 🟩] → With deadlines  
[Enemies 🔵] → Randomly move  
```

![Simulation View](screenshots/simulation_view.png)

**CSP Planning Logic Debug View**

![CSP Logic View](screenshots/csp_debug_overlay.png)

---

## 📜 License

This project is provided for **academic and educational use only**.
📘 No commercial license is granted.
