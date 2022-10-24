from abc import ABC, abstractmethod


class HighLevelPlanner(ABC):
    """
    Planner / Policy class that produces actions for a high level agent
    """
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def act(self, o):
        """
        Given an observation from the planning environment, produce a planning env action and a bool signal saying we are done planning or not
        """
        pass
    def generate_teacher_trajectory(self, planning_env):
        pass
    @abstractmethod
    def need_replan(self, curr_state, student_obs, teacher_trajectory):
        pass
