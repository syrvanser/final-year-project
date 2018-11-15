import math
import numpy as np

class MCTS():
    """
    Attributes:
        Nsa: An integer for visit count.
        Wsa: A float for the total action value.
        Qsa: A float for the mean action value.
        Psa: A float for the prior probability of reaching this node.
        action: A tuple(row, column) of the prior move of reaching this node.
        children: A list which stores child nodes.
        child_psas: A vector containing child probabilities.
        parent: A TreeNode representing the parent node.
    """
    def __init__(self, c_puct):
            """Initializes TreeNode with the initial statistics and data."""
            self.Nsa = 0 
            self.Wsa = 0.0
            self.Qsa = 0.0
            self.c_puct = c_puct
            

            
    def is_not_leaf(self):
            return len(self.children) > 0

    def select_child(self):
        """Selects a child node based on the AlphaZero PUCT formula.
        Returns:
            A child TreeNode which is the most promising according to PUCT.
        """

        highest_uct = 0
        highest_index = 0

        # Select the child with the highest Q + U value
        for idx, child in enumerate(self.children):
            uct = child.Qsa + child.Psa * c_puct * (
                    math.sqrt(self.Nsa) / (1 + child.Nsa))
            if uct > highest_uct:
                highest_uct = uct
                highest_index = idx

        return self.children[highest_index]