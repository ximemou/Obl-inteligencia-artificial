import random
from game import Directions
from util import manhattanDistance

MAX_NUMBER = float('inf')
MIN_NUMBER = float('-inf')
FOOD = -3
GHOST = -5
SCARED_GHOST = -5
CAPSULE = -15

def evaluationFunction(currentGameState):

  pacman_position = currentGameState.getPacmanPosition()
  score = currentGameState.getScore()
  foods = currentGameState.getFood().asList()
  number_of_capsules = len(currentGameState.getCapsules())
  ghosts = currentGameState.getGhostStates()
  scared_ghosts = []
  not_scared_ghosts = []

  if currentGameState.isLose() or currentGameState.isWin(): 
    return score

  for g in ghosts:
    if (g.scaredTimer):
      scared_ghosts.append(g.getPosition())
    else:
      not_scared_ghosts.append(g.getPosition())
  
  closest_food_distance = min(map(lambda food_position: manhattanDistance(pacman_position, food_position), foods))
  closest_ghost_distance = 999999
  closest_scared_ghost_distance = 0

  if(not_scared_ghosts):
    closest_ghost_distance = min(map(lambda ghost_position: manhattanDistance(pacman_position, ghost_position), not_scared_ghosts))
  
  if(scared_ghosts):
    closest_scared_ghost_distance = min(map(lambda ghost_position: manhattanDistance(pacman_position, ghost_position), scared_ghosts))
  
  score = score + (FOOD * closest_food_distance) + (CAPSULE * number_of_capsules) + (SCARED_GHOST * closest_scared_ghost_distance) + (GHOST * (1./closest_ghost_distance))
  return score

def getAction(gameState, d):
  global GHOSTS_NUMBER
  GHOSTS_NUMBER = gameState.getNumAgents() - 1
  agent_index = 0
  return (max_utility_pacman(gameState, agent_index, d))[0]

def max_utility_pacman(state, current_agent, d):
  max_utility_agents = MIN_NUMBER
  best_action = Directions.STOP

  for a in state.getLegalActions():
    a_utility = calculate_utility_ghosts(state.generateSuccessor(current_agent, a), current_agent + 1, d)
    if(a_utility > max_utility_agents):
      max_utility_agents = a_utility
      best_action = a

  return best_action, max_utility_agents

def calculate_utility_ghosts(state, current_agent, d):
  if(isTerminalState(state, current_agent, d)):
    return evaluationFunction(state)

  elif(isPacmanTurn(current_agent)):
    return (max_utility_pacman(state, 0, d - 1))[1]

  else:
    min_utility_ghosts = MAX_NUMBER
    for a in state.getLegalActions(current_agent):
      a_utility = calculate_utility_ghosts(state.generateSuccessor(current_agent, a), current_agent + 1, d)
      if(a_utility < min_utility_ghosts):
        min_utility_ghosts = a_utility
    return min_utility_ghosts

def isTerminalState(state, current_agent, d):
  return state.isWin() or state.isLose() or (d==1 and current_agent == GHOSTS_NUMBER)

def isPacmanTurn(current_agent): 
  return current_agent > GHOSTS_NUMBER