import pybullet as p
import pybullet_data
import time
import heapq

# Connect to PyBullet GUI
p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

# Maze layout: 0 = empty, 1 = wall
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
]

rows = len(maze)
cols = len(maze[0])

# Center camera to view whole maze
center_x = (cols - 1) / 2
center_y = (rows - 1) / 2
p.resetDebugVisualizerCamera(
    cameraDistance=7,
    cameraYaw=45,
    cameraPitch=-45,
    cameraTargetPosition=[center_x, center_y, 0]
)

# Build walls
for y in range(rows):
    for x in range(cols):
        if maze[y][x] == 1:
            wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
            wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0.3, 0.3, 0.3, 1])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_shape, baseVisualShapeIndex=wall_visual,
                              basePosition=[x, rows - y - 1, 0.5])

# Create robot cube
bot_size = 0.2
bot_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[bot_size]*3)
bot_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[bot_size]*3, rgbaColor=[1, 0, 0, 1])
start = (0, 4)
goal = (4, 0)
bot_pos = [start[0], rows - start[1] - 1, bot_size]
bot_id = p.createMultiBody(1, bot_col, bot_vis, basePosition=bot_pos)

# Create goal cube
goal_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[bot_size]*3, rgbaColor=[0, 1, 0, 1])
p.createMultiBody(0, -1, goal_vis, basePosition=[goal[0], rows - goal[1] - 1, bot_size])


# === A* Pathfinding ===
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def neighbors(pos):
    x, y = pos
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < cols and 0 <= ny < rows and maze[ny][nx] == 0:
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
            new_cost = cost_so_far[current] + 1
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(goal, next_node)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current

    # Reconstruct path
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from.get(node)
        if node is None:
            return []
    path.reverse()
    return path


# === Interpolated Movement ===
def move_bot_in_steps(start_pos, end_pos, step_size=0.05, speed_factor=1.0):
    x1, y1, z1 = start_pos
    x2, y2, z2 = end_pos
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
    num_steps = int(distance / (step_size * speed_factor))  # Adjust steps based on speed factor

    for i in range(1, num_steps + 1):
        interpolated_pos = [
            x1 + (x2 - x1) * i / num_steps,
            y1 + (y2 - y1) * i / num_steps,
            z1 + (z2 - z1) * i / num_steps
        ]
        p.resetBasePositionAndOrientation(bot_id, interpolated_pos, [0, 0, 0, 1])
        for _ in range(30):  # Step simulation a bit to visualize the movement
            p.stepSimulation()
            time.sleep(1 / 60)

# === Execute Path ===
path = astar(start, goal)
print("A* path:", path)

speed_factor = 4  # Adjust this variable to make the bot move slower or faster (1.0 is normal speed)

for step in path:
    target_pos = [step[0], rows - step[1] - 1, bot_size]
    move_bot_in_steps(bot_pos, target_pos, step_size=0.05, speed_factor=speed_factor)  # Smooth movement with smaller steps
    bot_pos = target_pos  # Update current position to the target position

input("Press Enter to exit...")
p.disconnect()
