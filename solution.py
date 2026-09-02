from typing import Self
import numpy as np

class Solution:

    def __init__(self, gnome: np.ndarray):
        self.gnome = gnome              # Encoded solution
        self.rank = None                # Front it belongs
        self.crowding_distance = None   # This is a value that measures how "different" is this solution among solutions
                                        # in the same front
        self.dominated_solutions = []   # This holds pointers to the solutions dominated by this solution
        self.dominated_by = 0           # This holds the number of times this solution has been dominated
        self.fitness = None

    def better_than(self, otherGnome: Self) -> bool: # Returns if a given solution is dominated by this solution
        itDominates = False

        if self.rank < otherGnome.rank:
            itDominates = True
        elif (self.rank == otherGnome.rank and
              self.crowding_distance > otherGnome.crowding_distance):
            itDominates = True

        return itDominates

    def dominates(self, otherGnome: Self) -> bool: # Returns if a given solution is dominated by this solution
        itDominates = False

        if ((self.fitness[0] >= otherGnome.fitness[0] and self.fitness[1] >= otherGnome.fitness[1] and self.fitness[2] <= otherGnome.fitness[2]) and
            (self.fitness[0] > otherGnome.fitness[0] or self.fitness[1] > otherGnome.fitness[1] or self.fitness[2] < otherGnome.fitness[2])):
            itDominates = True

        return itDominates