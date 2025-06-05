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
    Plan an optimal goal schedule minimizing total travel time while respecting deadlines,
    treating required and optional goals simultaneously like a CSP.

    Parameters:
        start: (x, y) start position
        goals: list of (x, y, deadline) — deadline==0 means optional
        grid: 2D grid of maze
        gridSize: parameter for astar (if needed)
        enemies: list of enemy positions (optional)

    Returns:
        list of (x, y) positions representing full flattened path through goals
    """

    fear_set = None
    memo = {}

    # Separate required and optional goals
    required_goals = [((x, y), d) for (x, y, d) in goals if d > 0]
    optional_goals = [((x, y), d) for (x, y, d) in goals if d == 0]

    def dfs(current_pos, time_so_far, remaining_required, remaining_optional, visited_optional):
        key = (
            current_pos,
            time_so_far,
            tuple(sorted(remaining_required)),
            tuple(sorted(remaining_optional)),
            tuple(sorted(visited_optional)),
        )
        if key in memo:
            return memo[key]

        # If no required or optional goals left, return empty path
        if not remaining_required and not remaining_optional:
            return [], 0

        best_path = []
        best_cost = float('inf')

        # Try visiting each required goal next (required goals must meet deadlines)
        for (goal_pos, deadline) in remaining_required:
            path = astar(current_pos, goal_pos, grid, gridSize, fear_set)
            if path is None:
                continue

            travel_time = len(path)
            arrival_time = time_so_far + travel_time

            # Skip if deadline violated
            if arrival_time > deadline:
                continue

            new_remaining_required = remaining_required.copy()
            new_remaining_required.remove((goal_pos, deadline))

            # Optional goals and visited optional remain unchanged here
            sub_path, sub_cost = dfs(goal_pos, arrival_time, new_remaining_required, remaining_optional, visited_optional)
            total_cost = travel_time + sub_cost

            if total_cost < best_cost:
                # Merge paths avoiding duplicates
                if sub_path and path[-1] == sub_path[0]:
                    combined_path = path + sub_path[1:]
                else:
                    combined_path = path + sub_path

                best_path = combined_path
                best_cost = total_cost

        # Try visiting optional goals if not already visited
        for (opt_pos, _) in remaining_optional:
            if opt_pos in visited_optional:
                continue  # skip if already visited

            path = astar(current_pos, opt_pos, grid, gridSize, fear_set)
            if path is None:
                continue

            travel_time = len(path)
            arrival_time = time_so_far + travel_time

            # Prune if visiting optional goal delays any required goal beyond deadline
            prune = False
            for (req_pos, req_deadline) in remaining_required:
                # estimate travel from optional goal to required goal (heuristic: Manhattan)
                est_travel = abs(opt_pos[0] - req_pos[0]) + abs(opt_pos[1] - req_pos[1])
                if arrival_time + est_travel > req_deadline:
                    prune = True
                    break
            if prune:
                continue

            new_visited_optional = visited_optional.copy()
            new_visited_optional.add(opt_pos)

            # We keep remaining_optional the same because optional goals can be revisited if wanted,
            # but we prevent revisiting via visited_optional
            sub_path, sub_cost = dfs(opt_pos, arrival_time, remaining_required, remaining_optional, new_visited_optional)
            # Add a small penalty for optional detours if desired, else zero
            optional_penalty = 0

            total_cost = travel_time + sub_cost + optional_penalty

            if total_cost < best_cost:
                if sub_path and path[-1] == sub_path[0]:
                    combined_path = path + sub_path[1:]
                else:
                    combined_path = path + sub_path

                best_path = combined_path
                best_cost = total_cost

        memo[key] = (best_path, best_cost)
        return memo[key]

    final_path, _ = dfs(start, 0, required_goals, optional_goals, set())
    return final_path


