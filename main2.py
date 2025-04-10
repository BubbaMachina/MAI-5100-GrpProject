import pybullet as p
import pybullet_data
import time
import heapq
import random

# === PARAMETERS ===
maze_rows = 10
maze_cols = 10
obstacle_prob = 0.2  # Probability of a wall per cell
num_goals = 3
num_agents = 2
bot_size = 0.2

# === SETUP ===
p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

# Camera centered
center_x = (maze_cols - 1) / 2
center_y = (maze_rows - 1) / 2
p.resetDebugVisualizerCamera(
    cameraDistance=10,
    cameraYaw=45,
    cameraPitch=-45,
    cameraTargetPosition=[center_x, center_y, 0]
)

# === RANDOM MAZE GENERATION ===
def generate_maze(rows, cols, obstacle_prob=0.2):
    maze = []
    for y in range(rows):
        row = []
        for x in range(cols):
            if random.random() < obstacle_prob:
                row.append(1)
            else:
                row.append(0)
        maze.append(row)
    return maze

maze = generate_maze(maze_rows, maze_cols, obstacle_prob)
start = (0, maze_rows - 1)
maze[start[1]][start[0]] = 0

# Place static walls
for y in range(maze_rows):
    for x in range(maze_cols):
        if maze[y][x] == 1:
            wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
            wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0.3, 0.3, 0.3, 1])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_shape, baseVisualShapeIndex=wall_visual,
                              basePosition=[x, maze_rows - y - 1, 0.5])

# === PLACE BOT ===
bot_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[bot_size]*3)
bot_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[bot_size]*3, rgbaColor=[1, 0, 0, 1])
bot_pos = [start[0], maze_rows - start[1] - 1, bot_size]
bot_id = p.createMultiBody(1, bot_col, bot_vis, basePosition=bot_pos)

# === PLACE GOALS ===
goal_positions = []
goal_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[bot_size]*3, rgbaColor=[0, 1, 0, 1])
while len(goal_positions) < num_goals:
    x, y = random.randint(0, maze_cols - 1), random.randint(0, maze_rows - 1)
    if maze[y][x] == 0 and (x, y) != start:
        goal_positions.append((x, y))
        p.createMultiBody(0, -1, goal_vis, basePosition=[x, maze_rows - y - 1, bot_size])

# Pick closest goal
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

goal = min(goal_positions, key=lambda g: heuristic(start, g))

# === PLACE TRAFFIC AGENTS ===
agent_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4]*3, rgbaColor=[0, 0, 1, 1])
traffic_agents = []
occupied_positions = set(goal_positions + [start])

while len(traffic_agents) < num_agents:
    x, y = random.randint(0, maze_cols - 1), random.randint(0, maze_rows - 1)
    if maze[y][x] == 0 and (x, y) not in occupied_positions:
        agent_id = p.createMultiBody(0, -1, agent_vis, basePosition=[x, maze_rows - y - 1, 0.4])
        traffic_agents.append([(x, y), agent_id])
        occupied_positions.add((x, y))

# === A* PATHFINDING ===
def neighbors(pos):
    x, y = pos
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze_cols and 0 <= ny < maze_rows and maze[ny][nx] == 0:
            yield (nx, ny)

def astar(start, goal):
    frontier = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for next_node in neighbors(current):
            if next_node in [a[0] for a in traffic_agents]:  # Avoid traffic agent positions
                continue
            new_cost = cost_so_far[current] + 1
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(goal, next_node)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current

    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from.get(node)
        if node is None:
            return []
    path.reverse()
    return path

# === MOVE BOT ===
def move_bot_in_steps(start_pos, end_pos, step_size=0.05, speed_factor=1.0):
    x1, y1, z1 = start_pos
    x2, y2, z2 = end_pos
    distance = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) ** 0.5
    num_steps = int(distance / (step_size * speed_factor))

    for i in range(1, num_steps + 1):
        interp_pos = [
            x1 + (x2 - x1) * i / num_steps,
            y1 + (y2 - y1) * i / num_steps,
            z1 + (z2 - z1) * i / num_steps
        ]
        p.resetBasePositionAndOrientation(bot_id, interp_pos, [0, 0, 0, 1])
        for _ in range(30):
            p.stepSimulation()
            time.sleep(1 / 60)

# === MOVE TRAFFIC AGENTS RANDOMLY ===
def move_agents_randomly():
    global traffic_agents
    updated_agents = []
    for (pos, aid) in traffic_agents:
        possible_moves = list(neighbors(pos))
        random.shuffle(possible_moves)
        moved = False
        for new_pos in possible_moves:
            if new_pos not in [a[0] for a in updated_agents] and maze[new_pos[1]][new_pos[0]] == 0:
                p.resetBasePositionAndOrientation(
                    aid,
                    [new_pos[0], maze_rows - new_pos[1] - 1, 0.4],
                    [0, 0, 0, 1]
                )
                updated_agents.append((new_pos, aid))
                moved = True
                break
        if not moved:
            updated_agents.append((pos, aid))  # No valid move
    traffic_agents = updated_agents

# === PLAN AND EXECUTE ===
print("Selected goal:", goal)
path = astar(start, goal)
print("Planned path:", path)

if path:
    for step in path:
        next_pos = [step[0], maze_rows - step[1] - 1, bot_size]
        move_bot_in_steps(bot_pos, next_pos, step_size=0.05, speed_factor=2.0)
        bot_pos = next_pos
        move_agents_randomly()  # Move traffic agents after each bot step

input("Press Enter to exit...")
p.disconnect()
