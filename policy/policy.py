"""Policy Template

This module includes all necessary features for PolicyGen class.
Module can be assigned as blue/red policy within CtF environment
for Capture the Flag environment.
    http://github.com/osipychev/missionplanner/

DOs/Denis Osipychev
    http://www.denisos.com
"""

class Policy:
    """Policy generator class for CtF env.
    
    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.
    
    Methods:
        initiate: Required method that runs everytime episode is initialized.
        gen_action: Required method to generate a list of actions.
    """
    
    def __init__(self):
        """Constuctor for policy class.
        
        This class can be used as a template for policy generator.

        Define:
            agent_list (list): list of all friendly units.
            free_map (np.array): 2d map of static environment (optional).
        
        """
        self.free_map = None
        self.agent_list = None
        
    def gen_action(self, agent_list, observation):
        """Action generation method.
        
        This is a required method that generates list of actions corresponding 
        to the list of units. 
        
        Args:
            agent_list (list): list of all friendly units.
            observation (np.array): 2d map of partially observable map.
            free_map (np.array): 2d map of static environment (optional).
            
        Returns:
            action_out (list): list of integers as actions selected for team.
        """
        raise NotImplementedError

    def initiate(self, free_map, agent_list):
        """Initiation method
        
        This method is called when the environment reset method is called.
        Any initialization or initiation should be included here
        The new static-map and agent list is given as parameter
        
        Args:
            agent_list (list): list of all friendly units.
            free_map (np.array): 2d map of static environment (optional).
        """
        self.free_map = free_map
        self.agent_list = agent_list
