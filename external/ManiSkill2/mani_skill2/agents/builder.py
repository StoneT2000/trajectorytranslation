import sapien.core as sapien
import yaml

import mani_skill2.agents as agent_zoo
from mani_skill2.agents.base_agent import AgentConfig, BaseAgent


def create_agent_from_config(
    config_path: str, scene: sapien.Scene, control_freq: int
) -> BaseAgent:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        config = AgentConfig(**config_dict)
    AgentClass = getattr(agent_zoo, config.agent_class)
    agent = AgentClass(config, scene, control_freq)
    return agent
