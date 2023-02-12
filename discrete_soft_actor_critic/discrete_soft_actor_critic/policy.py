from pathlib import Path
from .wrappers import EnvWrapper, ObsWrapper
from .network import DiscreteSAC, CollisionPredictor, TaskClassifer
import torch
import torch.nn.functional as F

from smarts.core.agent import Agent


def submitted_wrappers():
    wrappers = [
        ObsWrapper,
        EnvWrapper,
    ]
    return wrappers


class Policy(Agent):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.net = DiscreteSAC(100, 11, 9, 2).to(device)
        self.collision_predictor = CollisionPredictor(100, 11)
        self.classifier = TaskClassifer()

        self.load()
        
    def load(self):
        params = torch.load(Path(__file__).absolute().parents[0] / "SSM_1.pt", map_location=torch.device(self.device))
        params_c = torch.load(Path(__file__).absolute().parents[0] / "classifier_model.pt", map_location=torch.device(self.device))
        
        self.net.load_state_dict(params["net"])
        self.collision_predictor.load_state_dict(params["collision_predictor"])
        self.classifier.load_state_dict(params_c)
        
    @torch.no_grad()
    def act(self, obs):
        if obs is {}:
            return {}
        a = {}
        for agent_id in obs.keys():
            o = torch.from_numpy(obs[agent_id]).to(self.device).float().reshape(1, -1)
            classifier_input = o[:, 100:]
            o = o[:, :100]
            
            # classify task
            logits = self.classifier(classifier_input)
            z_map = F.one_hot(logits.argmax(-1), 9).reshape(1, -1).float()
            
            # collision prediction
            collision_prob = self.collision_predictor(o).mean(-1)
            
            p = self.net.P_net(o, z_map, collision_prob)
            a[agent_id] = p.argmax().item()
        return a
