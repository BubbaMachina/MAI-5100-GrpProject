from mazeGenerator import generateMaze, assignRandomDeadlines, generate_Agents
from graphics import generate_pybullet_maze, move_bot_in_steps, display_goal_deadlines, draw_path_lines, update_path_as_bot_moves, display_text_above_bot, rayCast
from agent import astar, greedyAstar, greedy_aStar_with_CSP
import pybullet as p
import math
import numpy as np
import random
# gridsize, obstacle is 1, # of goals(2), # of enemies(3)
# Player is #4
bot_size = 0.15
gridSize = 12
obst_prob = 0.1
num_agents = 7
num_goals = 3
maze1,goals1,startPos = generateMaze(gridSize,obst_prob,num_goals)

maze2 = generate_Agents(num_agents, maze1, gridSize)

goals1 = assignRandomDeadlines(goals1,gridSize,gridSize*2,0.3)
print("1 maze is:",maze1)
print("2 goals are",goals1)
print("3 starting position is",startPos)

# Generate the GUI maze
playerId,goals, traffic_agents = generate_pybullet_maze(maze1,gridSize,gridSize)



print("5 player id: ",playerId, "  Goals IDs are ", str(goals))

display_goal_deadlines(goals1,gridSize)
# Now Generate the Solutions

# a-Single Goal path planning
# path = astar(startPos,goals1[0],maze1,gridSize)
# b-Multi-Goal Greedy Path planning
# path = greedyAstar(startPos,maze1,gridSize)
path = greedy_aStar_with_CSP(startPos,goals1,maze1,gridSize)


print("6 Final Planned path:",path)
pathLines = draw_path_lines(path,gridSize)



def move_agents_randomly(agent_ids, maze, grid_size):
    for agent_id in agent_ids:
        pos, orn = p.getBasePositionAndOrientation(agent_id)
        x, y = int(pos[0]), grid_size - int(pos[1]) - 1

        for _ in range(10):  # Try up to 10 times
            dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and maze[ny][nx] == 0:
                new_pos = [nx, grid_size - ny - 1, 0.5]

                # Optional: face direction (not critical)
                angle = 0
                if dx == 1: angle = 180
                elif dx == -1: angle = -180
                elif dy == 1: angle = -90
                elif dy == -1: angle = 90
                new_orn = p.getQuaternionFromEuler([0, 0, math.radians(angle)])

                # Smooth move!
                move_bot_in_steps(agent_id, pos, new_pos, new_orn, step_size=0.05, speed_factor=3)
                break


# Code to interpolate solution path into pybullet movement
"""if path:
    bot_pos = p.getBasePositionAndOrientation(playerId)[0]
    print("7 bot position is ", bot_pos)
    text_id = None
    for i,step in enumerate(path):
        
        update_path_as_bot_moves(path,pathLines,i-1)
            
        next_pos = [step[0], gridSize - step[1] - 1, bot_size]
        
        #orientation stuffs
        future_x = next_pos[0]
        cur_x = bot_pos[0]
        future_y = next_pos[1]
        cur_y = bot_pos[1]

        if future_x > cur_x:
            angle = 180 # moving east

        elif future_x < cur_x:
            angle = -180 # moving west

        elif future_y > cur_y:
            angle = -90 #North

        elif future_y < cur_y:
            angle = 90
        else:
            angle = 0

        bot_orientation = p.getQuaternionFromEuler([0, 0, math.radians(angle)])
        #end orientation stuffs
        
        rayCast(bot_pos, playerId, traffic_agents)
        
        move_bot_in_steps(playerId,bot_pos, next_pos,bot_orientation, step_size=0.1, speed_factor=0.3)


        bot_pos = next_pos
        
        status_text = f"Time:{i}"
        text_id = display_text_above_bot(bot_pos, status_text, text_id)
        # move_agents_randomly()  # Move traffic agents after each bot step
        # if i>0:
        #     update_path_as_bot_moves(path,pathLines,i-1)
        move_agents_randomly(traffic_agents, maze1, gridSize)

        """

bot_pos = p.getBasePositionAndOrientation(playerId)[0]
bot_tile = (int(bot_pos[0]), gridSize - int(bot_pos[1]) - 1)
text_id = None

remaining_goals = goals1.copy()
step_counter = 0

while remaining_goals:
    # Recalculate path each loop (already done at the end of the last loop)
    if not path:
        print("No valid path to remaining goals.")
        break

    # Move one step
    next_step = path[0]
    next_pos = [next_step[0], gridSize - next_step[1] - 1, bot_size]

    dx = next_pos[0] - bot_pos[0]
    dy = next_pos[1] - bot_pos[1]
    angle = 180 if dx > 0 else -180 if dx < 0 else -90 if dy > 0 else 90 if dy < 0 else 0
    bot_orientation = p.getQuaternionFromEuler([0, 0, math.radians(angle)])

    # Perform ray detection BEFORE movement
    enemy_detected = rayCast(bot_pos, playerId, traffic_agents)
    if enemy_detected:
        # Update dynamic maze with new traffic positions
        dynamic_maze = np.copy(maze1)
        for agent_id in traffic_agents:
            pos, _ = p.getBasePositionAndOrientation(agent_id)
            x = int(round(pos[0]))
            y = gridSize - int(round(pos[1])) - 1
            if 0 <= x < gridSize and 0 <= y < gridSize:
                dynamic_maze[y][x] = 1

        # Recalculate path with updated maze
        path = greedy_aStar_with_CSP(bot_tile, remaining_goals, dynamic_maze, gridSize)

        # Redraw path
        if 'pathLines' in locals():
            for seg in pathLines:
                if seg is not None:
                    p.removeUserDebugItem(seg)
        pathLines = draw_path_lines(path, gridSize)
        continue  # Restart loop with new path

    # No enemy, safe to move
    move_bot_in_steps(playerId, bot_pos, next_pos, bot_orientation, step_size=0.1, speed_factor=0.3)
    bot_pos = next_pos
    bot_tile = (next_step[0], next_step[1])

    # Remove goal if reached
    remaining_goals = [g for g in remaining_goals if (g[0], g[1]) != bot_tile]

    # Update status
    text_id = display_text_above_bot(bot_pos, f"Time: {step_counter}", text_id)
    step_counter += 1

    # Move traffic agents randomly
    move_agents_randomly(traffic_agents, maze1, gridSize)

    # Recalculate path
    dynamic_maze = np.copy(maze1)
    for agent_id in traffic_agents:
        pos, _ = p.getBasePositionAndOrientation(agent_id)
        x = int(round(pos[0]))
        y = gridSize - int(round(pos[1])) - 1
        if 0 <= x < gridSize and 0 <= y < gridSize:
            dynamic_maze[y][x] = 1

    path = greedy_aStar_with_CSP(bot_tile, remaining_goals, dynamic_maze, gridSize)

    # Redraw path
    if 'pathLines' in locals():
        for seg in pathLines:
            if seg is not None:
                p.removeUserDebugItem(seg)
    pathLines = draw_path_lines(path, gridSize)


input("Press Enter to exit...")
p.disconnect()

