from mazeGenerator import generateMaze, assignRandomDeadlines
from graphics import generate_pybullet_maze, move_bot_in_steps, display_goal_deadlines, draw_path_lines, update_path_as_bot_moves, display_text_above_bot, rayCast
from agent import astar, greedyAstar, greedy_aStar_with_CSP
import pybullet as p
import math

# gridsize, obstacle is 1, # of goals(2), # of enemies(3)
# Player is #4
bot_size = 0.15
gridSize = 12
obst_prob = 0.2
num_agents = 5
num_goals = 3
maze1,goals1,startPos = generateMaze(gridSize,obst_prob,num_agents,num_goals)
goals1 = assignRandomDeadlines(goals1,gridSize,gridSize*2,0.3)
print("1 maze is:",maze1)
print("2 goals are",goals1)
print("3 starting position is",startPos)

# Generate the GUI maze
playerId,goals,traffic_agents_arr = generate_pybullet_maze(maze1,gridSize,gridSize)
print("5 player id: ",playerId, "  Goals IDs are ", str(goals), " Traffic agents are ", str(traffic_agents_arr))

display_goal_deadlines(goals1,gridSize)
# Now Generate the Solutions

# a-Single Goal path planning
# path = astar(startPos,goals1[0],maze1,gridSize)
# b-Multi-Goal Greedy Path planning
# path = greedyAstar(startPos,maze1,gridSize)
path = greedy_aStar_with_CSP(startPos,goals1,maze1,gridSize)

print("6 Final Planned path:",path)
pathLines = draw_path_lines(path,gridSize)
# Code to interpolate solution path into pybullet movement
if path:
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
        
        rayCast(bot_pos, playerId)
        
        move_bot_in_steps(playerId,bot_pos, next_pos,bot_orientation, step_size=0.1, speed_factor=0.3)
        bot_pos = next_pos
        
        status_text = f"Time:{i}"
        text_id = display_text_above_bot(bot_pos, status_text, text_id)
        # move_agents_randomly()  # Move traffic agents after each bot step
        # if i>0:
        #     update_path_as_bot_moves(path,pathLines,i-1)
input("Press Enter to exit...")
p.disconnect()

