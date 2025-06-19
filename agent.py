"""
    This file holds the decision logic
    for Bot and Adverserial/Netural Agents
    
    Define Bot & 
    
    Define A* and Heuristic Function
    
"""

###########################
# 1) Imports
###########################
import pybullet as p
import pybullet_data
import time
import heapq
import random

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


