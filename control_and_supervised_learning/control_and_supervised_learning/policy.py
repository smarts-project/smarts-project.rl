import numpy as np
from pathlib import Path
from typing import Any, Dict
import d3rlpy
from d3rlpy.dataset import MDPDataset
from eval import EnvWrapper
from smarts.core.agent import Agent

class Policy(Agent):
    """Policy class to be submitted by the user. This class will be loaded
    and tested during evaluation."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """
        # Load saved model and instantiate any needed objects.
        policy_name = Path(__file__).absolute().parents[0] / 'model_20000.pt'
        print('pln', policy_name)
        # TODO: Initialize the agent
        encoder = d3rlpy.models.encoders.VectorEncoderFactory([750, 750, 750])
        self.policy = d3rlpy.algos.DiscreteBC(
            use_gpu=False, batch_size=64, encoder_factory=encoder
        )
        observations = np.random.random((100, 53))  # Check whether the state dim can be passed from other places
        actions = np.random.random((100, 2))
        rewards = np.random.random(100)
        terminals = np.random.randint(2, size=100)
        demo_dataset_for_initialization = MDPDataset(observations, actions, rewards, terminals)
        self.policy.build_with_dataset(demo_dataset_for_initialization)
        self.policy.load_model(policy_name)  # path to the trained model
        self.agent_ids = set()
        self.actors = dict()

    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.
        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.
        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        raw_observations = obs

        # Use saved model to predict multi-agent action output given multi-agent SMARTS observation input.
        for agent, raw_obs in raw_observations.items():
            if agent not in self.agent_ids:
                self.actors[agent] = EnvWrapper(agent, self.policy)
                self.actors[agent].reset(raw_obs)
                self.agent_ids.add(agent)
            else:
                self.actors[agent].step(raw_obs)

        actions = {}
        wps_bias = {}
        for agent, raw_obs in raw_observations.items():
            wps_bias[agent] = self.actors[agent].cal_wps_bias()
        for agent, raw_obs in raw_observations.items():
            actions[agent] = self.actors[agent].rule_based_action(wps_bias)

        return actions
