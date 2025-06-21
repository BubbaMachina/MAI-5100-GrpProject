"""
    This file holds the decision logic
    for Bot and Adverserial/Netural Agents
    
    Define Bot & 
    
    Define A* and Heuristic Function
    
"""

###########################
# 1) Imports
###########################
import math
import pybullet as p
import pybullet_data
import time
import heapq
import random

from graphics import move_bot_in_steps

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def neighbors(pos,maze,gridSize):
    maze_rows = gridSize
    maze_cols = gridSize
    x, y = pos
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze_cols and 0 <= ny < maze_rows and maze[ny][nx] in (0,2,4):
            yield (nx, ny)

def astar(start, goal,maze,gridSize,exclude=None):
    if exclude is None:
        exclude = set()
    else:
        exclude = set(exclude)  # convert to set for fast lookups
    
    frontier = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        neigh = neighbors(current,maze,gridSize)
        # print(str(neigh))
        for next_node in neigh:
            # print("4.1 next node is",next_node)
            if next_node in exclude:  # Avoid traffic agent positions
                continue
            new_cost = cost_so_far[current] + 1
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(goal, next_node)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current

    path = []
    node = goal
    # print("4 now looking for path")
    while node != start:
        # print("node is " +str(node) +"- and start is "+ str(start))
        path.append(node)
        # print("came from is:"+str(came_from))
        # print(came_from[node])
        node = came_from.get(node)
        if node is None:
            # print("node is None")
            return []
    path.reverse()
    return path

def greedyAstar(start, maze, gridSize):
    def find_all_food(maze):
        goal_positions = []
        for y in range(gridSize):
            for x in range(gridSize):
                if maze[y][x] == 2:  # Goal state
                    goal_positions.append((x, y))
        return goal_positions

    current = start
    full_path = []

    while True:
        goal_positions = find_all_food(maze)
        if not goal_positions:
            break  # all food eaten

        # Find the closest food dot based on Manhattan distance
        closest_food = min(goal_positions, key=lambda f: heuristic(current, f))

        # Find actual shortest path to that dot using A*
        path_to_food = astar(current, closest_food, maze, gridSize)

        if not path_to_food:
            print(f"No path to food at {closest_food}")
            break

        # Add the path to the full path and simulate moving
        full_path.extend(path_to_food)

        # Move Pacman and eat food
        current = closest_food
        maze[current[1]][current[0]] = 0  # remove food (set to empty)

    return full_path

def greedy_aStar_with_CSP(start, goals, grid, gridSize, enemies=None):
    """
    Plan an optimal goal schedule minimizing total travel time while respecting deadlines.

    Parameters:
        start: (x, y) start position
        goals: list of (x, y, deadline) - deadline = 0 means no constraint, but goal is still mandatory
        grid: 2D grid representing the maze
        gridSize: cell size or related param for A* (if needed)
        enemies: list of enemy positions (optional)

    Returns:
        list of (x, y) positions representing full flattened path through all goals
    """
    from itertools import permutations

    fear_set = None  # Replace with actual fear radius logic if needed
    memo = {}

    required_goals = [((x, y), d) for (x, y, d) in goals]  # All are mandatory

    # Precompute pairwise shortest paths between all nodes (start + all goals)
    nodes = [start] + [pos for (pos, _) in required_goals]
    pairwise_paths = {}

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                src, dest = nodes[i], nodes[j]
                if (src, dest) not in pairwise_paths:
                    path = astar(src, dest, grid, gridSize, fear_set)
                    if path:
                        pairwise_paths[(src, dest)] = (path, len(path))
                    else:
                        pairwise_paths[(src, dest)] = (None, float('inf'))

    # Recursive DFS with memoization to explore all permutations
    def dfs(current_pos, time_so_far, remaining_goals):
        key = (current_pos, time_so_far, tuple(sorted(remaining_goals)))
        if key in memo:
            return memo[key]

        if not remaining_goals:
            return [], 0

        best_path = []
        best_cost = float('inf')

        for (goal_pos, deadline) in remaining_goals:
            path, travel_time = pairwise_paths.get((current_pos, goal_pos), (None, float('inf')))
            if path is None:
                continue

            arrival_time = time_so_far + travel_time
            if deadline > 0 and arrival_time > deadline:
                continue

            new_remaining = remaining_goals.copy()
            new_remaining.remove((goal_pos, deadline))

            sub_path, sub_cost = dfs(goal_pos, arrival_time, new_remaining)
            total_cost = travel_time + sub_cost

            if total_cost < best_cost:
                if sub_path and path[-1] == sub_path[0]:
                    combined_path = path + sub_path[1:]
                else:
                    combined_path = path + sub_path
                best_path = combined_path
                best_cost = total_cost

        memo[key] = (best_path, best_cost)
        return memo[key]

    final_path, _ = dfs(start, 0, required_goals)
    return final_path

def is_near(pos1, pos2, threshold=0.5):
    return (abs(pos1[0] - pos2[0]) < threshold and abs(pos1[1] - pos2[1]) < threshold)

def reflexive_bot_move(bot_id, traffic_agents_arr, gridSize, bot_size):
    # Get the current position of the bot
    bot_pos, bot_orient = p.getBasePositionAndOrientation(bot_id)
    bot_x, bot_y, bot_z = bot_pos
    
    # Find all nearby objects (traffic agents and walls)
    nearby_objects = []
    
    # Check traffic agents
    for agent in traffic_agents_arr:
        agent_pos, _ = p.getBasePositionAndOrientation(agent[1])
        if is_near(bot_pos, agent_pos):
            nearby_objects.append(agent_pos)
    
    # Check walls (using raycasting)
    ray_length = 1.0
    directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]  # front, back, right, left
    for dx, dy in directions:
        ray_end = [bot_x + dx * ray_length, bot_y + dy * ray_length, bot_z]
        ray_result = p.rayTest(bot_pos, ray_end)
        if ray_result[0][0] != -1:  # If hit something
            hit_pos = ray_result[0][3]
            nearby_objects.append(hit_pos)
    
    if not nearby_objects:
        return False  # No need to move reflexively
    
    # Find the nearest obstacle
    nearest_obstacle = None
    min_dist = float('inf')
    
    for obs_pos in nearby_objects:
        dist = ((bot_x - obs_pos[0])**2 + (bot_y - obs_pos[1])**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest_obstacle = obs_pos
    
    if nearest_obstacle:
        distance_x = nearest_obstacle[0] - bot_x
        distance_y = nearest_obstacle[1] - bot_y
        
        # Determine movement direction (away from obstacle)
        if abs(distance_x) > abs(distance_y):
            # Move away horizontally
            new_x = bot_x - 0.5 if distance_x > 0 else bot_x + 0.5
            new_y = bot_y  # Keep y-coordinate
        else:
            # Move away vertically
            new_y = bot_y - 0.5 if distance_y > 0 else bot_y + 0.5
            new_x = bot_x  # Keep x-coordinate
        
        # Ensure new position is within bounds
        new_x = max(bot_size, min(gridSize - 1 - bot_size, new_x))
        new_y = max(bot_size, min(gridSize - 1 - bot_size, new_y))
        
        # Calculate orientation
        angle = 0
        if new_x > bot_x:
            angle = 180  # east
        elif new_x < bot_x:
            angle = -180  # west
        elif new_y > bot_y:
            angle = -90  # north
        elif new_y < bot_y:
            angle = 90  # south
        
        new_orient = p.getQuaternionFromEuler([0, 0, math.radians(angle)])
        
        # Move smoothly to new position
        move_bot_in_steps(bot_id, bot_pos, [new_x, new_y, bot_z], new_orient, step_size=0.05, speed_factor=0.3)
        return True
    
    return False

