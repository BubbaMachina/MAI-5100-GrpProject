import pybullet as p
import pybullet_data
import time
import heapq
import random

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

def maze_y_to_pybullet_y(maze_y):
    return rows - maze_y - 1

# Center camera to view whole maze
center_x = (cols - 1) / 2
center_y = (rows - 1) / 2

p.resetDebugVisualizerCamera(
    cameraDistance=7,
    cameraYaw=180,
    cameraPitch=-100,
    cameraTargetPosition=[center_x, center_y, 0]
)

# Build walls
for y in range(rows):
    for x in range(cols):
        if maze[y][x] == 1:
            wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
            wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0.3, 0.3, 0.3, 1])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_shape, baseVisualShapeIndex=wall_visual,
                basePosition=[x, maze_y_to_pybullet_y(y), 0.5]) # Changed
            
# Define object sizes
object_size = 0.2

# Create robot cube
bot_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[object_size]*3)
bot_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[object_size]*3, rgbaColor=[1, 0, 0, 1])
bot_id = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=bot_col,
    baseVisualShapeIndex=bot_vis,
    basePosition=[0, maze_y_to_pybullet_y(0), object_size]  # X, Y, and height
)

print("rows is ", rows)
print("cols is ", cols)

# Create goal cube
goal_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[object_size]*3, rgbaColor=[0, 1, 0, 1])  # green
goal_id = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=-1,
    baseVisualShapeIndex=goal_vis,
    basePosition=[4, maze_y_to_pybullet_y(4), object_size]  # X, Y, and height
)

# Create obstacle cube
obstacle_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
obstacle_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[object_size]*3, rgbaColor=[0, 0, 1, 1])  # blue
obstacle_id = p.createMultiBody(
    baseMass=0,  # If > 0, gravity affects it
    baseCollisionShapeIndex=obstacle_col,
    baseVisualShapeIndex=obstacle_vis,
    basePosition=[2, maze_y_to_pybullet_y(2), object_size]  # X, Y, and height
)

# Create another obstacle cube
obstacle_col2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
obstacle_vis2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[object_size]*3, rgbaColor=[1, 0.5, 0, 1])  # orange
obstacle_id2 = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=obstacle_col2,
    baseVisualShapeIndex=obstacle_vis2,
    # OLD: basePosition=[3, maze_y_to_pybullet_y(3), object_size]
    basePosition=[2, maze_y_to_pybullet_y(3), object_size] 
)

for _ in range(60):  # about 1 second of simulation at 60 FPS
    p.stepSimulation()
    # time.sleep(1 / 60)

# def move_obstacle_randomly():
#     pos, _ = p.getBasePositionAndOrientation(obstacle_id)
#     pos2, _ = p.getBasePositionAndOrientation(obstacle_id2)

#     x, y, z = pos
#     x2, y2, z2 = pos2

#     # Choose a random direction (up/down/left/right)
#     dx, dy = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
#     dx2, dy2 = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])    
#     new_x = x + dx
#     new_y = y + dy
#     new_x2 = x2 + dx2
#     new_y2 = y2 + dy2
#     p.resetBasePositionAndOrientation(obstacle_id, [new_x, new_y, z], [0, 0, 0, 1])
#     p.resetBasePositionAndOrientation(obstacle_id2, [new_x2, new_y2, z2], [0, 0, 0, 1])

blue_step_count = 0
blue_direction = 1  # 1 for up, -1 for down
def move_obstacle_randomly():
    global blue_step_count, blue_direction
    
    # Blue cube (obstacle_id) - move up 5 steps, then down 5 steps in a loop
    pos, _ = p.getBasePositionAndOrientation(obstacle_id)
    x, y, z = pos
    
    # Move blue cube up/down based on pattern
    new_y = y + blue_direction
    blue_step_count += 1
    
    # Check if we need to change direction
    if blue_step_count >= 2:
        blue_direction *= -1  
        blue_step_count = 0
    
    p.resetBasePositionAndOrientation(obstacle_id, [x, new_y, z], [0, 0, 0, 1])
    
    # Orange cube (obstacle_id2) - keep random movement
    pos2, _ = p.getBasePositionAndOrientation(obstacle_id2)
    x2, y2, z2 = pos2
    dx2, dy2 = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])    
    new_x2 = x2 + dx2
    new_y2 = y2 + dy2
    p.resetBasePositionAndOrientation(obstacle_id2, [new_x2, new_y2, z2], [0, 0, 0, 1])


# main function:
# calculate the A* path to the goal
# === A* Pathfinding === Copied from the previous code
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

# and move the bot along the path

# bot_pos, _ = p.getBasePositionAndOrientation(bot_id)
# start_node_x = int(round(bot_pos[0]))
# start_node_y = rows - int(round(bot_pos[1])) - 1
# path = astar((start_node_x, start_node_y), (4, 4))

def move_bot_to_goal(bot_id, x_maze, y_maze):
    z = object_size
    p.resetBasePositionAndOrientation(bot_id, [x_maze, maze_y_to_pybullet_y(y_maze), z], [0, 0, 0, 1]) 
    for _ in range(30):
        p.stepSimulation()

# If the bot is close to the obstacle,
def is_near(pos1, pos2, threshold=0.2):
    return (abs(pos1[0] - pos2[0]) < threshold and abs(pos1[1] - pos2[1]) < threshold)

# move reflexively
def reflexive_bot_move(bot_id):
    # Get the current position of the bot and obstacles
    bot_pos, _ = p.getBasePositionAndOrientation(bot_id)
    obs1_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
    obs2_pos, _ = p.getBasePositionAndOrientation(obstacle_id2)

    # extract coordinates
    bot_x, bot_y, bot_z = bot_pos

    nearest_obstacle = None
    min_dist = float('inf')

    for obs_pos in [obs1_pos, obs2_pos]:
        # Calculate euclidean distance between bot and obstacle
        dist = ((bot_x - obs_pos[0]) ** 2 + (bot_y - obs_pos[1]) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            nearest_obstacle = obs_pos
    print("nearest_obstacle is ", nearest_obstacle)

    if nearest_obstacle:
        distance_x = nearest_obstacle[0] - bot_x
        distance_y = nearest_obstacle[1] - bot_y
        print("distance_x is ", distance_x)
        print("distance_y is ", distance_y)

        # example: bot is at (2, 0), obstacle at (1, 0)
        # distance_x = -1
        # distance_y = 0
        # abs(distance_x) > abs(distance_y) means move away horizontally
        # move back or forth?
        # +1 if obstacle is to the right, -1 if to the left
        if abs(distance_x) > abs(distance_y):
            # Move away horizontally
            new_x = bot_x - 1 if distance_x > 0 else bot_x + 1
            new_y = bot_y  # Keep y-coordinate
        else:
            # Move away vertically
            new_y = bot_y - 1 if distance_y > 0 else bot_y + 1
            new_x = bot_x  # Keep x-coordinate

            
        # Reset bot position to the new coordinates
        p.resetBasePositionAndOrientation(bot_id, [new_x, new_y, bot_z], [0, 0, 0, 1])

        return

# Actually move the objects (bot and obstacle)
# Get initial path
bot_pos, _ = p.getBasePositionAndOrientation(bot_id)
start_node_x = int(round(bot_pos[0]))
start_node_y = rows - int(round(bot_pos[1])) - 1
path = astar((start_node_x, start_node_y), (4, 4))


for step in path:
    move_obstacle_randomly()

    # Get the obstacles positions
    obs1_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
    obs2_pos, _ = p.getBasePositionAndOrientation(obstacle_id2)

    # get steps
    x, y = step
    next_bot_pos = [x, maze_y_to_pybullet_y(y), object_size]

    if is_near(next_bot_pos, obs1_pos) or is_near(next_bot_pos, obs2_pos):
        print("obstacle nearby. moving reflexively")
        reflexive_bot_move(bot_id)
    else:
        move_bot_to_goal(bot_id, x, y)

    for _ in range(15):
        p.stepSimulation()
        time.sleep(1 / 40)


input("Press Enter to exit...")
p.disconnect()