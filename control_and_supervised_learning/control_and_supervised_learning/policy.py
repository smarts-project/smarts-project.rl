import numpy as np
from pathlib import Path
from typing import Any, Dict
import d3rlpy
from d3rlpy.dataset import MDPDataset
from .eval import EnvWrapper
from smarts.core.agent import Agent

class Policy(Agent):
    """Policy class to be submitted by the user. This class will be loaded
    and tested during evaluation."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """
        # Load saved model and instantiate any needed objects.
        policy_name = Path(__file__).absolute().parent / 'model_20000.pt'
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
        self.actors = EnvWrapper(model=self.policy)
        self._initialized = False

    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.
        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.
        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        # Use saved model to predict multi-agent action output given multi-agent SMARTS observation input.
        if not self._initialized:
            self.actors.reset(obs)
            self._initialized = True
        else:
            self.actors.step(obs)

        actions = self.actors.rule_based_action()
        return actions