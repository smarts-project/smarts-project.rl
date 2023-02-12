from pathlib import Path

import torch
import torch.nn.functional as F

from smarts.core.agent import Agent

from .network import CollisionPredictor, DiscreteSAC, TaskClassifer
from .wrappers import EnvWrapper, ObsWrapper


class Policy(Agent):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.net = DiscreteSAC(100, 11, 9, 2).to(device)
        self.collision_predictor = CollisionPredictor(100, 11)
        self.classifier = TaskClassifer()
        self.load()
        self.obs_wrapper = ObsWrapper()
        self.env_wrapper = EnvWrapper(obs_wrapper=self.obs_wrapper)

    def load(self):
        params = torch.load(
            Path(__file__).absolute().parents[0] / "SSM_1.pt",
            map_location=torch.device(self.device),
        )
        params_c = torch.load(
            Path(__file__).absolute().parents[0] / "classifier_model.pt",
            map_location=torch.device(self.device),
        )

        self.net.load_state_dict(params["net"])
        self.collision_predictor.load_state_dict(params["collision_predictor"])
        self.classifier.load_state_dict(params_c)

    @torch.no_grad()
    def act(self, obs):

        # Reset
        # -----
        if obs.steps_completed == 1:
            self.env_wrapper.reset()
            self.obs_wrapper.reset()

        # Pre-process
        # -----------
        wrapped_obs = self.obs_wrapper.observation(obs)

        # Compute actions
        # ---------------
        o = torch.from_numpy(wrapped_obs).to(self.device).float().reshape(1, -1)
        classifier_input = o[:, 100:]
        o = o[:, :100]

        # classify task
        logits = self.classifier(classifier_input)
        z_map = F.one_hot(logits.argmax(-1), 9).reshape(1, -1).float()

        # collision prediction
        collision_prob = self.collision_predictor(o).mean(-1)

        p = self.net.P_net(o, z_map, collision_prob)
        a = p.argmax().item()

        # Post-process
        # ------------
        wrapped_act = self.env_wrapper.step(a)
        self.obs_wrapper.step()

        return wrapped_act
