import math

import torch
from torch.distributions import kl_divergence
import torch.nn.functional as F
from torch.nn import Module
import numpy as np

class ACFlow(Module):

    def __init__(self):
        super.__init__()

class Flow(Module):
    def __init__(self, n_concepts, n_tasks, layer_cfg = [], affine_hids = [256, 256]):
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.layer_cfg = layer_cfg
        self.affine_hids = affine_hids
        self.transform = Transform(n_concepts, n_tasks, affine_hids, layer_cfg)
        self.prior = Prior()

    def forward(self, x, b, m):
        x_u, x_o = self.preprocess(x, b, m)
        z_u, logdet = self.transform.forward(x_u, x_o, b, m)
        prior_ll = self.prior.logp(z_u, x_o, b, m)
        log_likel = prior_ll + logdet

        return log_likel

    def inverse(self, x, b, m):
        _, x_o = self.preprocess(x, b, m)
        z_u = self.prior.sample(x_o, b, m)
        x_u, _ = self.transform.inverse(z_u, x_o, b, m)
        x_sam = self.postprocess(x_u, x, b, m)

        return x_sam

    def mean(self, x, b, m):
        _, x_o = self.preprocess(x, b, m)
        z_u = self.prior.mean(x_o, b, m)
        x_u, _ = self.transform.inverse(z_u, x_o, b, m)
        x_mean = self.postprocess(x_u, x, b, m)

        return x_mean

    def cond_forward(self, x, y, b, m):
        x_u, x_o = self.preprocess(x, b, m)
        c = torch.concat([F.one_hot(y, self.n_concepts), x_o], dim=1)
        z_u, logdet = self.transform.forward(x_u, c, b, m)
        prior_ll = self.prior.logp(z_u, c, b, m)
        log_likel = prior_ll + logdet

        return log_likel

    def cond_inverse(self, x, y, b, m):
        _, x_o = self.preprocess(x, b, m)
        c = torch.concat([F.one_hot(y, self.n_concepts), x_o], dim=1)
        z_u = self.prior.sample(c, b, m)
        x_u, _ = self.transform.inverse(z_u, c, b, m)
        x_sam = self.postprocess(x_u, x, b, m)

        return x_sam

    def cond_mean(self, x, y, b, m):
        _, x_o = self.preprocess(x, b, m)
        c = torch.concat([F.one_hot(y, self.n_concepts), x_o], dim=1)
        z_u = self.prior.mean(c, b, m)
        x_u, _ = self.transform.inverse(z_u, c, b, m)
        x_mean = self.postprocess(x_u, x, b, m)

        return x_mean

    def preprocess(self, x, b, m):
        x_o = x * m * b
        x_u = x * m * (1 - b)
        query = m * (1 - b)
        ind = torch.argsort(query, dim=1, descending=True)
        x_u = torch.gather(x_u, 1, ind.expand_as(x_u))

        return x_u, x_o
    
    def postprocess(self, x_u, x, b, m):
        query = m * (1 - b)
        ind = torch.argsort(query, dim=1, descending=True)
        ind = torch.argsort(ind, dim=1)
        x_u = torch.gather(x_u, 1, ind.expand_as(x_u))
        sam = x_u * query + x * (1 - query)

        return sam
    
class BaseTransform(Module):
    def __init__(self):
        super.__init__()

    def forward(self, x, c, b, m):
        raise NotImplementedError()

    def inverse(self, z, c, b, m):
        raise NotImplementedError()
    
class Affine(BaseTransform):
    def __init__(self, n_concepts, n_tasks, affine_hids):
        super.__init__()
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.affine_hids = affine_hids

        layers = []
        for i, h in enumerate(self.affine_hids):
            layers.append(torch.nn.Linear(self.n_concept * 3 + self.n_tasks if i == 0 else self.affine_hids[i-1], h))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(self.affine_hids[-1], self.n_concepts * 2))
    
        torch.nn.init.zeros_(layers[-1].weight)

        self.net = torch.nn.Sequential(*layers)

    def get_params(self, c, b, m):
        h = torch.concat([c, b, m], dim=1)
        params = self.net(h)
        shift, scale = torch.split(params, len(params) / 2, dim=1)
        
        query = m * (1-b)   
        _, order = torch.sort(query, descending = True)

        t = torch.diag_embed(query)
        t = t.gather(1, order.unsqueeze(-1).expand(-1, -1, query.size(-1)))
        t = t.transpose(1, 2)
        
        scale = torch.einsum('nd,ndi->ni', scale, t)
        shift = torch.einsum('nd,ndi->ni', shift, t)

        return shift, scale
    
    def forward(self, x, c, b, m):
        shift, scale = self.get_params(c, b, m)
        z = torch.mul(x, torch.exp(scale)) + shift
        ldet = torch.sum(scale, dim=1)

        return z, ldet

    def inverse(self, z, c, b, m):
        shift, scale = self.get_params(c, b, m)
        x = torch.divide(z-shift, torch.exp(scale))
        ldet = -1 * torch.sum(scale, dim=1)

        return x, ldet
    

class Coupling2(BaseTransform):
    def __init__(self, n_concepts, n_tasks, affine_hids):
        super.__init__()
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.affine_hids = affine_hids

        layers = []
        for i, h in enumerate(self.affine_hids):
            layers.append(torch.nn.Linear(self.n_concept * 3 + self.n_tasks if i == 0 else self.affine_hids[i-1], h))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(self.affine_hids[-1], self.n_concepts * 2))
    
        torch.nn.init.zeros_(layers[-1].weight)

        self.net = torch.nn.Sequential(*layers)

    def get_params(self, c, b, m):
        h = torch.concat([c, b, m], dim=1)
        params = self.net(h)
        shift, scale = torch.split(params, len(params) / 2, dim=1)
        
        query = m * (1-b)   
        _, order = torch.sort(query, descending = True)

        t = torch.diag_embed(query)
        t = t.gather(1, order.unsqueeze(-1).expand(-1, -1, query.size(-1)))
        t = t.transpose(1, 2)
        
        scale = torch.einsum('nd,ndi->ni', scale, t)
        shift = torch.einsum('nd,ndi->ni', shift, t)

        return shift, scale
    
    def forward(self, x, c, b, m):
        shift, scale = self.get_params(c, b, m)
        z = torch.mul(x, torch.exp(scale)) + shift
        ldet = torch.sum(scale, dim=1)

        return z, ldet

    def inverse(self, z, c, b, m):
        shift, scale = self.get_params(c, b, m)
        x = torch.divide(z-shift, torch.exp(scale))
        ldet = -1 * torch.sum(scale, dim=1)

        return x, ldet
    
class LeakyReLU(BaseTransform):
    def __init__(self):
        super.__init__()
        log_alpha = torch.nn.Parameter(torch.tensor(5.0))
        self.alpha = torch.sigmoid(log_alpha)

    def forward(self, x, c, b, m):
        query = m * (1-b) # [B, d]
        sorted_query, _ = torch.sort(query, dim=-1, descending = True)
        num_negative = torch.sum((x < 0.).float() * sorted_query, axis=1)
        ldet = num_negative * torch.log(self.alpha)
        z = torch.maximum(x, self.alpha * x)

        return z, ldet

    def inverse(self, z, c, b, m):
        query = m * (1-b) # [B, d]
        sorted_query, _ = torch.sort(query, dim=-1, descending = True)
        num_negative = torch.sum((z < 0.).float() * sorted_query, axis=1)
        ldet = -1. * num_negative * torch.log(self.alpha)
        x = torch.minimum(z, z / self.alpha)
        return x, ldet

class LULinear(BaseTransform):
    def __init__(self, n_concepts, n_tasks, linear_rank, linear_hids):
        super.__init__()
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.linear_rank = n_concepts if linear_rank <= 0 else linear_rank
        self.linear_hids = linear_hids

        np_w = np.eye(n_concepts).astype("float32")
        self.w = torch.nn.Parameter(torch.tensor(np_w))
        self.b = torch.nn.Parameter(torch.tensor(np.zeros(n_concepts)))
        
        wnn = []
        bnn = []

        for i, h in enumerate(linear_hids):
            wnn.append(torch.nn.Linear(self.n_concept * 3 + self.n_tasks if i == 0 else linear_hids[i-1], h))
            wnn.append(torch.nn.Tanh())
            
            bnn.append(torch.nn.Linear(self.n_concept * 3 + self.n_tasks if i == 0 else linear_hids[i-1], h))
            bnn.append(torch.nn.Tanh())

        wnn.append(torch.nn.Linear(self.affine_hids[-1], self.n_concepts * 2 * self.linear_rank))
        torch.nn.init.zeros_(wnn[-1].weight)
        self.wnn = torch.nn.Sequential(*wnn)

        bnn.append(torch.nn.Linear(self.affine_hids[-1], self.n_concepts))
        torch.nn.init.zeros_(bnn[-1].weight)
        self.bnn = torch.nn.Sequential(*bnn)

    
class TransLayer(BaseTransform):
    def __init__(self, n_concepts, n_tasks, affine_hids, layer_cfg):
        super(Transform, self).__init__()
        self.transformations = []
        for name in layer_cfg:
            if name == "AF":
                self.transformation.append(Affine(n_concepts, n_tasks, affine_hids))
            elif name == "CP2":
                self.transformation.append(Coupling2(n_concepts, n_tasks, affine_hids))
            elif name == "LR":
                self.transformation.append(LeakyReLU())
            elif name == "ML":
                self.transformation.append((n_concepts, n_tasks, affine_hids))

    def forward(self, x, c, b, m):
        logdet = 0.
        for transformation in self.transformation:
            x, ldet = transformation(x, c, b, m)
            logdet = logdet + ldet

        return x, logdet

    def inverse(self, z, c, b, m):
        logdet = 0.
        for transformation in reversed(self.transformations):
            z, ldet = transformation.inverse(z, c, b, m)
            logdet = logdet + ldet

        return z, logdet
    
class Transform(BaseTransform):
    def __init__(self, n_concepts, n_tasks, affine_hids, layer_cfg, transformations):
        super(Transform, self).__init__()
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.affine_hids = affine_hids
        self.layer_cfg = layer_cfg

        self.transformations = []
        for name in transformations:
            m = self.create_transformation(name)
            self.transformation.append(m)

    def forward(self, x, c, b, m):
        logdet = 0.
        for transformation in self.transformation:
            x, ldet = transformation(x, c, b, m)
            logdet = logdet + ldet

        return x, logdet

    def inverse(self, z, c, b, m):
        logdet = 0.
        for transformation in reversed(self.transformations):
            z, ldet = transformation.inverse(z, c, b, m)
            logdet = logdet + ldet

        return z, logdet
    
    def create_transformation(self, name):
        if name == "AF":
            return Affine(self.n_concepts, self.n_tasks, self.affine_hids)
        elif name == "CP2":
            return Coupling2(self.n_concepts, self.n_tasks, self.affine_hids)
        elif name == "LR":
            return LeakyReLU(self.n_concepts, self.n_tasks, self.affine_hids)
        elif name == "ML":
            return LULinear(self.n_concepts, self.n_tasks, self.affine_hids)
        elif name == "TransLayer":
            return TransLayer(self.n_concepts, self.n_tasks, self.affine_hids)

