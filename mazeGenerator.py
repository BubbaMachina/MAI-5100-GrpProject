import random
import numpy as np

"""
    This program randomly generates a maze
    Gridsize determines the length & Height of the grid (assumed to be square)
    obst_prob is the Obstacle probability
    0 is empty, 1 is an obstacle
    2 is for the goals, and numGoals counts the amount
    3 is for agents, and numAgents counts
    4 is the player
    
"""

# Generatae a maze of 0s and 1s for wall or space. 2 for goal, 3s for start(s)
def generateMaze(dim,obst_prob=0.2,numAgents=1,numGoals=1):
  n = dim # nxn matrix, number of dimensions
  rand_matrix = np.zeros((n,n),dtype=int) #nxn matrix of zeros
  goals=[]
  start = None
  obst_norm = obst_prob*100 #normalized to 100 %
  
  #Assign walls to the grid
  for j in range(n):
    for k in range(n):
        if(random.randint(0,100) < obst_norm):
            rand_matrix[j][k] = 1

  
  # insert goals randomly
  for i in range(numGoals):
    ny=random.randint(0,n-1)
    nx=random.randint(0,n-1)
    rand_matrix[ny][nx] = 2
    goals.append((nx,ny))
  
  # Assign enemy agents
  for i in range(numAgents):
    rand_matrix[random.randint(0,n-1)][random.randint(0,n-1)] = 3
    
  # insert player randomly
  py = random.randint(0,n-1)
  px = random.randint(0,n-1)
  rand_matrix[py][px] = 4
  start = (px,py)
  
  return rand_matrix,goals,start

# Generates random deadlines and appends them to each goal's tuple
def assignRandomDeadlines(goals, minDeadline=15, maxDeadline=30, zeroDeadlineProb=0.3):
    """
    Assigns a deadline to each goal in the list.
    
    Args:
        goals (list of (x, y)): Goal positions.
        minDeadline (int): Minimum deadline value (inclusive) for constrained goals.
        maxDeadline (int): Maximum deadline value (inclusive) for constrained goals.
        zeroDeadlineProb (float): Probability of assigning an unconstrained deadline (0).

    Returns:
        list of (x, y, deadline): Each goal with its deadline.
    """
    deadline_goals = []
    for x, y in goals:
        if random.random() < zeroDeadlineProb:
            deadline = 0  # unconstrained
        else:
            deadline = random.randint(minDeadline, maxDeadline)
        deadline_goals.append((x, y, deadline))
    return deadline_goals
