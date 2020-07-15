from PacmanEnvAbs import PacmanEnvAbs
from util import manhattanDistance
import numpy as np

class PacmanEnv(PacmanEnvAbs):

    def _get_reward(self, pacman_successor):#pacman_successor is the state after pacman moved but before the other agent moved

        score = super(PacmanEnv, self)._get_reward(pacman_successor)

        observation = self._get_observations()
        pacman_position = self.get_elements(6, observation)[0]

        foods = self.get_elements(2, observation)
        capsules = self.get_elements(3, observation)
        ghosts = self.get_elements(4, observation)
        scared_ghosts = self.get_elements(5, observation)

        foods_distance_one = list(filter(lambda x: manhattanDistance(pacman_position, x) == 1, foods))
        capsules_distance_one = list(filter(lambda x: manhattanDistance(pacman_position, x) == 1, capsules))
        scared_ghosts_distance_one = list(filter(lambda x: manhattanDistance(pacman_position, x) == 1, scared_ghosts))

        food_for_pacman = len(foods_distance_one) + len(capsules_distance_one) + len(scared_ghosts_distance_one)
        if(food_for_pacman == 0):
            food_for_pacman = -1

        if pacman_successor.isLose() or pacman_successor.isWin():
            return score

        score = score + food_for_pacman  
        return score

    def get_elements(self, type_of_element, observation):
        elements_positions = []
        for row in range(len(observation)):
            for column in range(len(observation[0])):
                if observation[row][column] == type_of_element:
                    elements_positions.append((row, column))
        return elements_positions

    def _get_observations(self):
        return super(PacmanEnv, self)._get_observations()

    def flatten_obs(self, s):
        return super(PacmanEnv, self).flatten_obs(s)