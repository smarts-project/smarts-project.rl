import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import copy

class MLP(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, layers_size:List[int] = [256]*2,
        activation: nn.Module = nn.ReLU, output_activation: Optional[nn.Module] = None
    ):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, layers_size[0]), activation()]
        for i in range(len(layers_size) - 1):
            layers += [
                nn.Linear(layers_size[i], layers_size[i+1]),
                activation()
            ]
        layers.append(nn.Linear(layers_size[-1], output_dim))
        
        if output_activation is not None:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    


# ================================================ Soft Module

class TaskSpecObsEncoder(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.n_layer = nn.Sequential(
            nn.Linear(5,  5),
            nn.LayerNorm(5),
            nn.Tanh(),
        )
        self.l_layer = nn.Sequential(
            nn.Linear(25,  10),
            nn.LayerNorm(10),
            nn.Tanh(),
        )
        self.mix_layer = nn.Linear(55, h_dim)
    def forward(self, o):
        batch_size = o.shape[0]
        n_info = o[:, :25].reshape((batch_size, 5, 5))
        l_info = o[:, 25:].reshape((batch_size, 3, 25))
        n_enc = self.n_layer(n_info).reshape((batch_size, -1))
        l_enc = self.l_layer(l_info).reshape((batch_size, -1))
        return F.relu(self.mix_layer(torch.cat([n_enc, l_enc], -1))), o[:, -32:-29]
    
class SoftModuleLayer(nn.Module):
    def __init__(self, n_modules=4, h_dim=128):
        super().__init__()
        self.n = n_modules
        self.nets = nn.ModuleList([
            nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU()) for _ in range(n_modules)
        ])
    
    def forward(self, x, probs):
        assert len(x.shape) in [2, 3]
        if len(x.shape) == 2:
            y = torch.stack([self.nets[i](x) for i in range(self.n)], 1)
        else:
            y = torch.stack([self.nets[i](x[:, i]) for i in range(self.n)], 1)
        weighted_y = probs@y
        return weighted_y
    
class BaseNetwork(nn.Module):
    def __init__(self, input_dim=82, output_dim=11, n_layers=4, n_modules=4, h_dim=128, task_spec_obs_encoder=False):
        super().__init__()
        self.n_layers = n_layers
        self.obs_encode_layer = MLP(input_dim, h_dim, [h_dim], output_activation=nn.ReLU) \
            if not task_spec_obs_encoder else TaskSpecObsEncoder(h_dim)
        self.module_blocks = nn.ModuleList([SoftModuleLayer(n_modules, h_dim) for _ in range(n_layers)])
        self.output_layer = nn.Linear(h_dim, output_dim)
        
    def encode_obs(self, x):
        return self.obs_encode_layer(x)
    
    def forward(self, x_enc, probs, goal_info, collision_prob):
        '''
        x_enc: (batch_size, h_dim)
        probs: (n_layers-1, batch_size, n_modules, n_modules)
        '''
        for i in range(self.n_layers - 1):
            x_enc = self.module_blocks[i](x_enc, probs[i])
        x_enc = x_enc.sum(1).reshape((x_enc.shape[0], 4, -1))
        g = torch.ones((x_enc.shape[0], 4, 1), device = x_enc.device)
        g[:, :3] = (1 - goal_info.unsqueeze(-1))
        g[:, -1] = collision_prob.squeeze()
        x_enc *= g
        return self.output_layer(x_enc.reshape((x_enc.shape[0], -1)))
    
class RoutingNetwork(nn.Module):
    def __init__(self, task_dim=7, h_dim=128, n_layers=3, n_modules=4) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_modules = n_modules
        self.task_encode_layer = nn.Linear(task_dim, h_dim)
        self.task_info = None
        
        self.probs_layers = nn.ModuleList([nn.Linear(h_dim, n_modules**2) for _ in range(n_layers)])
        self.fullc_layers = nn.ModuleList([nn.Linear(n_modules**2, h_dim) for _ in range(n_layers)])
        
    def forward(self, task_id, x_enc):
        self.task_info = self.task_encode_layer(task_id) * x_enc
        
        probs = torch.empty((self.n_layers-1, task_id.shape[0], self.n_modules**2), device=task_id.device)
        for i in range(self.n_layers-1):
            if i == 0:
                probs[0] = self.probs_layers[0](self.task_info)
            else:
                probs[i] = self.probs_layers[i](self.fullc_layers[i-1](F.relu(probs[i-1])) * self.task_info)
        probs = F.softmax(probs.reshape(self.n_layers-1, task_id.shape[0], self.n_modules, self.n_modules), -1)
        return probs
    
class SoftModuleBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        task_dim,
        output_dim,
        h_dim = 128,
        n_layers = 4,
        n_modules = 4,
        task_spec_obs_encoder = False,
    ):
        super().__init__()
        self.base_net = BaseNetwork(input_dim, output_dim, n_layers, n_modules, h_dim, task_spec_obs_encoder)
        self.rouring_net = RoutingNetwork(task_dim, h_dim, n_layers, n_modules)
    
    def forward(self, x, z, collision_prob):
        x_enc, goal_info = self.base_net.encode_obs(x)
        probs = self.rouring_net(z, x_enc)
        y = self.base_net(x_enc, probs, goal_info, collision_prob)
        return y
        
class DuelingQNet(nn.Module):
    def __init__(self, o_dim, a_dim, z_map_dim=9):
        super().__init__()
        self.softmodule_block_map = SoftModuleBlock(o_dim, z_map_dim, 128, 128, 4, 4, True)
        self.V_layer = nn.Linear(128, 1)
        self.A_layer = MLP(128, a_dim, [128])
        
    def forward(self, o, z_map, collision_prob):
        feature = self.softmodule_block_map(o, z_map, collision_prob)
        value = self.V_layer(feature)
        advantage = self.A_layer(feature)
        advantage -= advantage.mean(-1, keepdim=True)
        return advantage + value
    
class PolicyNet(nn.Module):
    def __init__(self, o_dim, a_dim, z_map_dim=9):
        super().__init__()
        self.softmodule_block_map = SoftModuleBlock(o_dim, z_map_dim, a_dim, 128, 4, 4, True)

    def forward(self, o, z_map, collision_prob):
        logits = self.softmodule_block_map(o, z_map, collision_prob)
        return F.softmax(logits, -1)
    
class DiscreteSAC(nn.Module):
    def __init__(self, o_dim, a_dim, z_map_dim=9, n_critic = 1):
        super(DiscreteSAC, self).__init__()
        self.n_critic = n_critic
        self.a_dim = a_dim

        self.Q_net = nn.ModuleList([DuelingQNet(o_dim, a_dim, z_map_dim) for _ in range(n_critic)])
        self.P_net = PolicyNet(o_dim, a_dim, z_map_dim)
        
        self.temperature = 1.0
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            nn.init.zeros_(module.bias)
            
    def need_grad(self, module: nn.Module, need: bool):
        for p in module.parameters():
            p.requires_grad = need
        
    def forward(self, o, z_map, collision_prob):
        batch_size = o.shape[0]
        Qs = torch.empty((self.n_critic, batch_size, self.a_dim)).to(o.device)
        for i in range(self.n_critic):
            Qs[i, ...] = self.Q_net[i](o, z_map, collision_prob)
        Q = Qs.min(0).values
        P = self.P_net(o, z_map, collision_prob)
        return Q, P
    
    def cal_Q(self, o, z_map, collision_prob):
        batch_size = o.shape[0]
        Qs = torch.empty((self.n_critic, batch_size, self.a_dim)).to(o.device)
        for i in range(self.n_critic):
            Qs[i, ...] = self.Q_net[i](o, z_map, collision_prob)
        return Qs
    
    
class CollisionPredictor(nn.Module):
    def __init__(self, s_dim=100, a_dim=11):
        super().__init__()
        self.net = MLP(s_dim, a_dim, [256], output_activation=nn.Sigmoid)
        self.apply(self._init_weights)
        
        self.net_targ = copy.deepcopy(self.net)
        for p in self.net_targ.parameters():
            p.requires_grad = False
        self.opt = torch.optim.Adam(self.net.parameters(), lr = 5E-4)
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            nn.init.zeros_(module.bias)
    def _update_target_parameters(self):
        for p, p_targ in zip(self.net.parameters(), self.net_targ.parameters()):
            p_targ.data.lerp_(p.data, 0.005)
    def update(self, s, a, R, s2, pi):
        batch_size = s.shape[0]
        # TD-target
        with torch.no_grad():
            y = R.clone().reshape(batch_size, 1)
            Q2 = self.net_targ(s2)
            # Q2 = self.net(s2)
            idx = torch.where(R == -1)
            y[idx] = 0.9 * ((Q2*pi).sum(-1, keepdims=True))[idx]
        Q = self.forward(s)[
            torch.arange(batch_size).reshape(batch_size, 1),
            a.long().reshape(batch_size, 1)
        ]
        loss = F.mse_loss(Q, y)
        # loss = (-y*(Q+1e-8).log()-(1-y)*(1-Q+1e-8).log()).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.detach().item()
    def forward(self, s):
        return self.net(s)
    
    
    
class TaskSpecObsEncoder2(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.n_layer = nn.Sequential(
            nn.Linear(5,  5),
            nn.LayerNorm(5),
            nn.Tanh(),
        )
        self.l_layer = nn.Sequential(
            nn.Linear(70,  25),
            nn.LayerNorm(25),
            nn.Tanh(),
        )
        self.mix_layer = nn.Linear(100, h_dim)
    def forward(self, o):
        batch_size = o.shape[0]
        n_info = o[:, :25].reshape((batch_size, 5, 5))
        l_info = o[:, 25:].reshape((batch_size, 3, 70))
        n_enc = self.n_layer(n_info).reshape((batch_size, -1))
        l_enc = self.l_layer(l_info).reshape((batch_size, -1))
        return F.relu(self.mix_layer(torch.cat([n_enc, l_enc], -1)))
    
class TaskClassifer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = TaskSpecObsEncoder2(128)
        self.net = MLP(128, 9, [128]*3)
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
        
    def forward(self, obs):
        return self.net(self.encoder(obs))