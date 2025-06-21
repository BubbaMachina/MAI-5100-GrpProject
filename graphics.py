import pybullet as p
import pybullet_data
import time
import heapq
import random
import math

"""
    This section of code is responsible for converting the mazeGenerator into a pybullet Render
    
"""
# Setup pybullet and camera alone
def setupPybullet(maze_rows,maze_cols):
    # First setup pybullet physics and plane
    p.connect(p.GUI)
    # p.setGravity(0, 0, -9.81)
    p.setGravity(0, 0, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    # Next the Camera should be centered
    center_x = (maze_cols - 1) / 2
    center_y = (maze_rows - 1) / 2
    p.resetDebugVisualizerCamera(
        cameraDistance=10,
        cameraYaw=45,
        cameraPitch=-45,
        cameraTargetPosition=[center_x, center_y, 0]
    )
    
    
# Generate a pybullet maze
# First walls, then goals then agents
def generate_pybullet_maze(maze,maze_rows,maze_cols):
    setupPybullet(maze_rows, maze_cols)
    bot_size = 0.2
    plane = [0.5,0.5,0.12]
    goals = []
    player_id = None
    traffic_agents = []
    
    for y in range(maze_rows):
        for x in range(maze_cols):
            # 1-Place a wall as a cube
            if maze[y][x] == 1:
                wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
                wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0.3, 0.3, 0.3, 1])
                p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_shape, baseVisualShapeIndex=wall_visual,
                                basePosition=[x, maze_rows - y - 1, 0.5])

            # 2-Place a goal cube
            elif maze[y][x] == 2:
                goal_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=plane, rgbaColor=[0, 1, 0, 1])
                goal_id = p.createMultiBody(0, -1, goal_vis, basePosition=[x, maze_rows - y - 1, plane[2]])
                goals.append(goal_id)
            
            # 3-Place Enemy agents
            elif maze[y][x] == 3:
                agent_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4]*3, rgbaColor=[0, 0, 1, 1])
                agent_id = p.createMultiBody(0, -1, agent_vis, basePosition=[x, maze_rows - y - 1, 0.4])
                traffic_agents.append([(x, y), agent_id])
                # occupied_positions.add((x, y))
            
            
            # 4-Place the Player
            elif maze[y][x] == 4:
                
                bot_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[bot_size]*3)
                bot_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[bot_size]*3, rgbaColor=[1, 0, 0, 1])
                # bot_id = p.createMultiBody(1, bot_col, bot_vis, basePosition=[x,maze_rows-y-1,bot_size])  
                bot_id = p.loadURDF("husky/husky.urdf", basePosition=[x, maze_rows - y - 1, 0.2],globalScaling=0.9)
                player_id = bot_id
                
                # create ground tile
                ground_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=plane, rgbaColor=[1, 1, 0, 1])
                p.createMultiBody(0, -1, ground_vis, basePosition=[x, maze_rows - y - 1, plane[2]])
            # For ground tiles
            else:
                ground_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=plane, rgbaColor=[1, 1, 1, 1])
                p.createMultiBody(0, -1, ground_vis, basePosition=[x, maze_rows - y - 1, plane[2]])
    
    return player_id,goals,traffic_agents
                
# Bot movement

def move_bot_in_steps(bot_id, start_pos, end_pos, orientation, step_size=0.05, speed_factor=1.0):
    x1, y1, z1 = start_pos
    x2, y2, z2 = end_pos
    distance = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5
    num_steps = max(1, int(distance / step_size))  # Fine-grained interpolation

    # Calculate time per step based on desired speed factor
    base_step_duration = 1 / 60.0  # base: 60 FPS
    step_duration = base_step_duration / speed_factor  # faster speed_factor = less sleep

    for i in range(1, num_steps + 1):
        interpolated_pos = [
            x1 + (x2 - x1) * i / num_steps,
            y1 + (y2 - y1) * i / num_steps,
            z1 + (z2 - z1) * i / num_steps
        ]
        p.resetBasePositionAndOrientation(bot_id, interpolated_pos, orientation)
        p.stepSimulation()
        time.sleep(step_duration)

def display_goal_deadlines(goal_tuples, maze_rows):
    for x, y, deadline in goal_tuples:
        # Ensure centered, valid, and visible placement
        px = x -0.3
        py = maze_rows - y - 1 
        pz = 0.5

        text = str(int(deadline)) if deadline > 0 else "Any"
        print("goal text is-", text)
        texty = "dline"
        # Render safely
        p.addUserDebugText(
            text,
            [px, py, pz],
            textColorRGB=[1, 0, 0],
            textSize=1.0,
            lifeTime=0
        )

def draw_path_lines(path, maze_rows, z_height=0.3, color=[1, 0.3, 0.3], lineWidth=2):
    """
    Draw the entire path as persistent debug lines and return their IDs.

    Args:
        path (list of (x, y)): The path coordinates.
        maze_rows (int): Number of rows in the maze (used to convert Y coords).
        z_height (float): Height to draw the lines at.
        color (list): RGB color for the path lines.
        lineWidth (float): Thickness of lines.

    Returns:
        list: IDs of the debug lines created.
    """
    if not path or len(path) < 2:
        return []

    segment_ids = []
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]

        line_id = p.addUserDebugLine(
            [x1, maze_rows - y1 - 1, z_height],
            [x2, maze_rows - y2 - 1, z_height],
            lineColorRGB=color,
            lineWidth=lineWidth,
            lifeTime=0  # persistent
        )
        segment_ids.append(line_id)
    return segment_ids


def update_path_as_bot_moves(path, segment_ids, current_step_index):
    """
    Removes the debug line segment that corresponds to the bot passing current_step_index.

    Args:
        path (list of (x, y)): The path coordinates.
        segment_ids (list): List of debug line IDs from draw_path_lines.
        current_step_index (int): Index of the path segment just reached by the bot.

    Returns:
        None
    """
    # Defensive: only remove if index valid and segment exists
    if 0 <= current_step_index < len(segment_ids):
        p.removeUserDebugItem(segment_ids[current_step_index])
        # Optionally, set the id to None to mark it's removed
        segment_ids[current_step_index] = None

def display_text_above_bot(bot_position, text, previous_text_id=None):
    """
    Displays a text label above the bot in the simulation.

    Args:
        bot_position (tuple): (x, y, z) world position of the bot.
        text (str): Text to display.
        previous_text_id (int or None): ID of the previously shown text to remove.

    Returns:
        int: ID of the new debug text, to be reused in the next frame.
    """
    if previous_text_id is not None:
        p.removeUserDebugItem(previous_text_id)

    x, y, z = bot_position
    position_3d = [x, y, z + 1]  # Place text slightly above bot

    text_id = p.addUserDebugText(
        text,
        position_3d,
        textColorRGB=[1, 0, 0],
        textSize=1.2,
        lifeTime=0
    )

    return text_id

def rayCast(position, bot_id):

    print("Bot Position:", position, "bot_id:", bot_id)

    aabb_min, aabb_max = p.getAABB(bot_id)
    height = aabb_max[2] - aabb_min[2]
    print("bot height ", height)

    ray_height = aabb_max[2] + 0.2   #needed to put the ray slightly above the bot to prevent self interference.

    ray_length = 5

    ray_results = []

    directions = [
        ('+X', ray_length, 0, [1, 0, 0]),
        ('-X', -ray_length, 0, [1, 0, 0]),
        ('+Y', 0, ray_length, [0, 1, 0]),
        ('-Y', 0, -ray_length, [0, 1, 0]) 
    ]

    for name, dx, dy, color in directions:
        ray_start = [position[0], position[1], ray_height]  # [position[0], position[1], position[2]]
        ray_end   = [position[0] + dx, position[1] + dy, ray_height] #[position[0] + dx, position[1] + dy, position[2]]
        result = p.rayTest(ray_start, ray_end)[0]


        ray_results.append((result[0], result[2], result[3])) #store the ID, hit fraction and v3 coords of object that was hit.
        p.addUserDebugLine(ray_start, ray_end, color, lineWidth=2.0, lifeTime=0.1)

    print("ray results:", ray_results, "\n")