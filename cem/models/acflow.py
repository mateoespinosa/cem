import math
import torch

import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import tensorflow as tf

from torch.distributions import kl_divergence
from torch.nn import Module
from torchmetrics import Accuracy

from torch.utils.data import Dataset

class ACFlow(pl.LightningModule):

    def __init__(self, n_concepts, n_tasks, layer_cfg, affine_hids,  linear_rank, linear_hids, transformations, prior_units, prior_layers, prior_hids, n_components, optimizer = None, learning_rate = None, weight_decay = None, momentum = None, lambda_xent = 1, lambda_nll = 1, float_type = "float32"):
        super().__init__()
        self.n_concepts = n_concepts
        n_tasks = n_tasks if n_tasks > 1 else 2 
        self.n_tasks = n_tasks
        self.flow = Flow(n_concepts, n_tasks, layer_cfg, affine_hids, linear_rank, linear_hids, transformations,  prior_units, prior_layers, prior_hids, n_components, float_type)
        self.lambda_xent = lambda_xent
        self.lambda_nll = lambda_nll
        self.xent_loss = torch.nn.CrossEntropyLoss() if n_tasks > 1 else torch.nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task = "multiclass" if n_tasks > 2 else "binary", num_classes = n_tasks if n_tasks > 1 else 2)
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.float_type = float_type

    def flow_forward(self, x, b, m, y = None, forward = True, task = "classify"):
        B = x.shape[0]
        d = self.n_concepts
        N = self.n_tasks
        x = torch.tile(torch.unsqueeze(x, dim = 1), [1, N, 1])
        x = torch.reshape(x, [B * N, d])
        b = torch.tile(torch.unsqueeze(b, dim = 1), [1, N, 1])
        b = torch.reshape(b, [B * N, d])
        m = torch.tile(torch.unsqueeze(m, dim = 1), [1, N, 1])
        m = torch.reshape(x, [B * N, d])
        if(y == None):
            if(task == "classify"):
                y = torch.tile(torch.unsqueeze(torch.arange(N), dim = 0), [B, 1])
                y = y.to(x.device)
                forward = True
            elif(task == "sample"):
                y = torch.randint(0, self.n_tasks, [B*N])
                y = y.to(x.device)
                forward = False
        if(y.shape != (B*N,)):
            if(y.shape == (B,)):
                y = torch.tile(torch.unsqueeze(y, dim = 1), [1, N])
                y = torch.reshape(y, [B * N])
            elif(y.shape == (B, N)):
                y = torch.reshape(y, [B * N])
            else:
                raise ValueError(f"y should have shape ({B}) or ({B},{N}). Instead y is of shape {y.shape}")
        # log p(x_u | x_o, y)
        if forward:
            logp = self.flow.cond_forward(x, y, b, m)
            # logits
            logits = torch.reshape(logp, [B,N])
            return logits
        else:
            sample = self.flow.cond_inverse(x,y,b,m)
            sample = torch.reshape(sample, [B,N,d])
            return sample

    def forward(self, x, b, m, y):
        # if x.requires_grad:
        #     import pdb
        #     pdb.set_trace()
        # log p(x_u | x_o, y)
        logpu = self.flow_forward(x, b, m, None, task = "classify")
        # log p(x_o | y)
        logpo = self.flow_forward(x, b * (1-b), b, None, task = "classify")
        
        sam = self.flow_forward(x, b, m, None, task = "sample")

        # sample p(x_u | x_o, y)
        if y is not None:
            cond_sam = self.flow_forward(x, b, m, y, forward = False)
        else:
            cond_sam = None
        # sample p(x_u | x_o, y) based on predicted y
        pred = torch.argmax(logpo, dim=1)
        pred_sam = self.flow_forward(x, b, m, pred, forward = False)

        return logpu, logpo, sam, cond_sam, pred_sam

    def training_step(self, batch, batch_idx):
        
        x, b, m, y = batch['x'], batch['b'], batch['m'], batch['y']
        class_weights = torch.tensor(np.array(batch.get('class_weights', [1. for _ in range(self.n_tasks)])).astype(self.float_type)).to(x.device)
        class_weights /= torch.sum(class_weights)
        class_weights = torch.log(class_weights)
        class_weights = torch.tile(torch.unsqueeze(class_weights, dim = 0), [x.shape[0], 1])

        logpu, logpo, _, _, _ = self(x,b,m,y)

        logits = logpu + logpo
        xent = self.xent_loss(logits, y)

        loglikel = torch.logsumexp(logpu + logpo + class_weights, dim = 1) - torch.logsumexp(logpo + class_weights, dim = 1)
        nll = torch.mean(-loglikel)
        
        loss = xent * self.lambda_xent + self.lambda_nll * nll

        pred = torch.argmax(logits, dim=1)
        acc = self.accuracy(pred, y)

        result = {"loss": loss.detach(), "accuracy": acc.detach(), "nll": nll.detach()}

        for name, val in result.items():
            self.log(name, val, prog_bar=("accuracy" in name))

        return {"loss": loss, "accuracy": acc, "nll": nll}

    def validation_step(self, batch, batch_idx):

        x, b, m, y = batch['x'], batch['b'], batch['m'], batch['y']
        class_weights = torch.tensor(np.array(batch.get('class_weights', [1. for _ in range(self.n_tasks)])).astype(self.float_type)).to(x.device)
        class_weights /= torch.sum(class_weights)
        class_weights = torch.log(class_weights)
        class_weights = torch.tile(torch.unsqueeze(class_weights, dim = 0), [x.shape[0], 1])

        logpu, logpo, _, _, _ = self(x,b,m,y)

        logits = logpu + logpo
        xent = self.xent_loss(logits, y)

        loglikel = torch.logsumexp(logpu + logpo + class_weights, dim = 1) - torch.logsumexp(logpo + class_weights, dim = 1)
        nll = torch.mean(-loglikel)
        
        loss = xent * self.lambda_xent + self.lambda_nll * nll

        pred = torch.argmax(logits, dim=1)
        acc = self.accuracy(pred, y)

        result = {"loss": loss.detach(), "accuracy": acc.detach(), "nll": nll.detach()}

        for name, val in result.items():
            self.log(name, val, prog_bar=("accuracy" in name))

        return {"loss": loss, "accuracy": acc, "nll": nll}

    def test_step(self, batch, batch_idx):
        
        x, b, m, y = batch['x'], batch['b'], batch['m'], batch['y']
        class_weights = torch.tensor(np.array(batch.get('class_weights', [1. for _ in range(self.n_tasks)])).astype(self.float_type)).to(x.device)
        class_weights /= torch.sum(class_weights)
        class_weights = torch.log(class_weights)
        class_weights = torch.tile(torch.unsqueeze(class_weights, dim = 0), [x.shape[0], 1])

        logpu, logpo, _, _, _ = self(x,b,m,y)

        logits = logpu + logpo

        loglikel = torch.logsumexp(logpu + logpo + class_weights, dim = 1) - torch.logsumexp(logpo + class_weights, dim = 1)
        nll = torch.mean(-loglikel)
        
        pred = torch.argmax(logits, dim=1)
        acc = self.accuracy(pred, y)

        result = {"accuracy": acc.detach(), "nll": nll.detach()}

        for name, val in result.items():
            self.log(name, val, prog_bar=("accuracy" in name))

        return result

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.flow.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.flow.parameters()),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            verbose=True,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }

class Flow(Module):
    def __init__(self, n_concepts, n_tasks, layer_cfg, affine_hids, linear_rank, linear_hids, transformations, prior_units, prior_layers, prior_hids, n_components, float_type = "float64"):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.layer_cfg = layer_cfg
        self.affine_hids = affine_hids
        self.transform = Transform(n_concepts, n_tasks, affine_hids, layer_cfg, linear_rank, linear_hids, transformations, float_type)
        self.prior = AutoReg(
            n_concepts = n_concepts, 
            n_tasks = n_tasks, 
            prior_units = prior_units, 
            prior_layers = prior_layers, 
            prior_hids = prior_hids, 
            n_components = n_components,
            float_type = float_type
        )
        self.float_type = float_type

    def forward(self, x, b, m):
        x_u, x_o = self.preprocess(x, b, m)
        import pdb
        pdb.set_trace()
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
        c = torch.concat([F.one_hot(y, self.n_tasks), x_o], dim=1)
        z_u, logdet = self.transform.forward(x_u, c, b, m)
        prior_ll = self.prior.logp(z_u, c, b, m)
        log_likel = prior_ll + logdet

        return log_likel

    def cond_inverse(self, x, y, b, m):
        _, x_o = self.preprocess(x, b, m)
        c = torch.concat([F.one_hot(y, self.n_tasks), x_o], dim=1)
        z_u = self.prior.sample(c, b, m)
        x_u, _ = self.transform.inverse(z_u, c, b, m)
        x_sam = self.postprocess(x_u, x, b, m)

        return x_sam

    def cond_mean(self, x, y, b, m):
        _, x_o = self.preprocess(x, b, m)
        c = torch.concat([F.one_hot(y, self.n_tasks), x_o], dim=1)
        z_u = self.prior.mean(c, b, m)
        x_u, _ = self.transform.inverse(z_u, c, b, m)
        x_mean = self.postprocess(x_u, x, b, m)

        return x_mean

    def preprocess(self, x, b, m):
        x_o = x * m * b
        x_u = x * m * (1 - b)
        query = m * (1 - b)
        ind = torch.argsort(query, dim=1, descending=True, stable=True)
        x_u = torch.gather(x_u, 1, ind)

        return x_u, x_o
    
    def postprocess(self, x_u, x, b, m):
        query = m * (1 - b)
        ind = torch.argsort(query, dim=1, descending=True, stable=True)
        ind = torch.argsort(ind, dim=1, stable=True)
        x_u = torch.gather(x_u, 1, ind)
        sam = x_u * query + x * (1 - query)

        return sam
    
class BaseTransform(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, c, b, m):
        raise NotImplementedError()

    def inverse(self, z, c, b, m):
        raise NotImplementedError()
    
class Affine(BaseTransform):
    def __init__(self, n_concepts, n_tasks, affine_hids):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.affine_hids = affine_hids

        layers = []
        for i, h in enumerate(self.affine_hids):
            layers.append(torch.nn.Linear(self.n_concepts * 3 + self.n_tasks if i == 0 else self.affine_hids[i-1], h))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(self.affine_hids[-1], self.n_concepts * 2))
    
        torch.nn.init.zeros_(layers[-1].weight)

        self.net = torch.nn.Sequential(*layers)

    def get_params(self, c, b, m):
        h = torch.concat([c, b, m], dim=1)
        params = self.net(h)
        shift, scale = torch.split(params, self.n_concepts, dim=1)
        
        query = m * (1-b)   
        _, order = torch.sort(query, descending = True, stable=True)

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
        super().__init__()
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.affine_hids = affine_hids

        layers = []
        for i, h in enumerate(self.affine_hids):
            layers.append(torch.nn.Linear(self.n_concepts * 3 + self.n_tasks if i == 0 else self.affine_hids[i-1], h))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(self.affine_hids[-1], self.n_concepts * 2))
    
        torch.nn.init.zeros_(layers[-1].weight)

        self.net = torch.nn.Sequential(*layers)

    def get_params(self, c, b, m):
        h = torch.concat([c, b, m], dim=1)
        params = self.net(h)
        shift, scale = torch.split(params, self.n_concepts, dim=1)
        
        query = m * (1-b)   
        _, order = torch.sort(query, descending = True, stable=True)

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
    def __init__(self, float_type):
        super().__init__()
        self.log_alpha = torch.nn.Parameter(torch.tensor(5.0)).to(torch.float64 if float_type == "float64" else torch.float32)
        self.float_type = float_type

    def forward(self, x, c, b, m):
        query = m * (1-b) # [B, d]
        sorted_query, _ = torch.sort(query, dim=-1, descending = True, stable=True)
        num_negative = torch.sum((x < 0.).to(torch.float64 if self.float_type == "float64" else torch.float32) * sorted_query, dim=1)
        alpha = torch.sigmoid(self.log_alpha)
        ldet = num_negative * torch.log(alpha)
        z = torch.maximum(x, alpha * x)

        return z, ldet

    def inverse(self, z, c, b, m):
        query = m * (1-b) # [B, d]
        sorted_query, _ = torch.sort(query, dim=-1, descending = True, stable=True)
        num_negative = torch.sum((z < 0).to(torch.float64 if self.float_type == "float64" else torch.float32) * sorted_query, dim=1)
        alpha = torch.sigmoid(self.log_alpha)
        ldet = -1. * num_negative * torch.log(alpha)
        x = torch.minimum(z, z / alpha)
        return x, ldet

class LULinear(BaseTransform):
    def __init__(self, n_concepts, n_tasks, linear_rank, linear_hids, float_type = "torchfloat64"):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.linear_rank = n_concepts if linear_rank <= 0 else linear_rank
        self.linear_hids = linear_hids

        np_w = np.eye(n_concepts).astype(float_type)
        self.w = torch.nn.Parameter(torch.tensor(np_w))
        self.b = torch.nn.Parameter(torch.tensor(np.zeros(n_concepts).astype(float_type)))
        
        wnn = []
        bnn = []

        for i, h in enumerate(linear_hids):
            wnn.append(torch.nn.Linear(self.n_concepts * 3 + self.n_tasks if i == 0 else linear_hids[i-1], h))
            wnn.append(torch.nn.Tanh())
            
            bnn.append(torch.nn.Linear(self.n_concepts * 3 + self.n_tasks if i == 0 else linear_hids[i-1], h))
            bnn.append(torch.nn.Tanh())

        wnn.append(torch.nn.Linear(self.linear_hids[-1], self.n_concepts * 2 * self.linear_rank))
        torch.nn.init.zeros_(wnn[-1].weight)
        self.wnn = torch.nn.Sequential(*wnn)

        bnn.append(torch.nn.Linear(self.linear_hids[-1], self.n_concepts))
        torch.nn.init.zeros_(bnn[-1].weight)
        self.bnn = torch.nn.Sequential(*bnn)

    def get_params(self, c, b, m):
        B = c.shape[0]
        d = self.n_concepts
        r = self.linear_rank
        r = d if r <= 0 else r
        h = torch.concat([c, b, m], dim=1)
        wc = self.wnn(h)
        wc1, wc2 = torch.split(wc, self.n_concepts * self.linear_rank, dim=1)
        wc1 = torch.reshape(wc1, [B,d,r])
        wc2 = torch.reshape(wc2, [B,r,d])
        wc = torch.matmul(wc1, wc2)
        bc = self.bnn(h)
        weight = wc + self.w
        bias = bc + self.b
        # reorder
        query = m * (1-b)
        order = torch.argsort(query, descending = True, stable=True)
        order = torch.unsqueeze(order, dim = -1)
        order = torch.tile(order, dims = (1,1,d))
        t = torch.diag_embed(query)
        t = torch.gather(t, 1, order)

        weight = torch.matmul(torch.matmul(t, weight), torch.permute(t, (0,2,1)))
        bias = torch.squeeze(torch.matmul(t, torch.unsqueeze(bias, dim = -1)), dim = -1)
        
        return weight, bias

    def get_LU(self, W, b, m):
        d = self.n_concepts
        U = torch.triu(W)
        L = torch.eye(d, device=W.device) + W - U
        A = torch.matmul(L, U)

        # add a diagnal part
        query = m * (1 - b)
        sorted_tensor, _ = torch.sort(1 - query, dim = 1, descending = False)
        diag = torch.diag_embed(sorted_tensor)
        U = U + diag

        return A, L, U

    def forward(self, x, c, b, m):
        weight, bias = self.get_params(c, b, m)
        A, L, U = self.get_LU(weight, b, m)    
        ldet = torch.sum(torch.log(torch.abs(torch.diagonal(U, offset = 0, dim1=-2, dim2=-1))), dim=1)
        z = torch.einsum('ai,aik->ak', x, A) + bias    

        return z, ldet

    def inverse(self, z, c, b, m):
        weight, bias = self.get_params(c, b, m)
        A, L, U = self.get_LU(weight, b, m)
        ldet = -1 * torch.sum(torch.log(torch.abs(torch.diagonal(U, offset = 0, dim1=-2, dim2=-1))), dim=1)
        Ut = torch.permute(U, (0, 2, 1))
        Lt = torch.permute(L, (0, 2, 1))
        zt = torch.unsqueeze(z - bias, dim = -1)
        sol = torch.linalg.solve_triangular(Ut, zt, upper=False)
        x = torch.linalg.solve_triangular(Lt, sol, upper=True)
        x = torch.squeeze(x, dim=-1)

        return x, ldet

    
class TransLayer(BaseTransform):
    def __init__(self, n_concepts, n_tasks, affine_hids, layer_cfg, linear_rank, linear_hids, float_type = "torch64"):
        super().__init__()
        self.transformations = torch.nn.ModuleList([])
        for name in layer_cfg:
            if name == "AF":
                self.transformations.append(Affine(n_concepts, n_tasks, affine_hids))
            elif name == "CP2":
                self.transformations.append(Coupling2(n_concepts, n_tasks, affine_hids))
            elif name == "LR":
                self.transformations.append(LeakyReLU(float_type))
            elif name == "ML":
                self.transformations.append(LULinear(n_concepts, n_tasks, linear_rank, linear_hids, float_type))
        self.float_type = float_type

    def forward(self, x, c, b, m):
        logdet = torch.zeros(x.shape[0], dtype = torch.float64 if self.float_type == "float64" else torch.float32).to(x.device)
        for transformation in self.transformations:
            x, ldet = transformation(x, c, b, m)
            logdet = logdet + ldet
            assert logdet.shape == ldet.shape

        return x, logdet

    def inverse(self, z, c, b, m):
        logdet = torch.zeros(z.shape[0],  dtype = torch.float64 if self.float_type == "float64" else torch.float32).to(z.device)
        for transformation in reversed(self.transformations):
            z, ldet = transformation.inverse(z, c, b, m)
            logdet = logdet + ldet
            assert logdet.shape == ldet.shape

        return z, logdet
    
class Transform(BaseTransform):
    def __init__(self, n_concepts, n_tasks, affine_hids, layer_cfg, linear_rank, linear_hids, transformations, float_type = "float64"):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.affine_hids = affine_hids
        self.layer_cfg = layer_cfg
        self.linear_rank = linear_rank
        self.linear_hids = linear_hids
        self.float_type = float_type

        self.transformations = torch.nn.ModuleList([])
        for name in transformations:
            m = self.create_transformation(name)
            self.transformations.append(m)


    def forward(self, x, c, b, m):
        logdet = torch.zeros(x.shape[0], dtype = torch.float64 if self.float_type == "float64" else torch.float32).to(x.device)

        for transformation in self.transformations:
            x, ldet = transformation(x, c, b, m)
            logdet = logdet + ldet
            assert logdet.shape == ldet.shape

        return x, logdet

    def inverse(self, z, c, b, m):
        logdet = torch.zeros(z.shape[0], dtype = torch.float64 if self.float_type == "float64" else torch.float32).to(z.device)
        for transformation in reversed(self.transformations):
            z, ldet = transformation.inverse(z, c, b, m)
            logdet = logdet + ldet
            assert logdet.shape == ldet.shape

        return z, logdet
    
    def create_transformation(self, name):
        if name == "AF":
            return Affine(self.n_concepts, self.n_tasks, self.affine_hids)
        elif name == "CP2":
            return Coupling2(self.n_concepts, self.n_tasks, self.affine_hids)
        elif name == "LR":
            return LeakyReLU(self.float_type)
        elif name == "ML":
            return LULinear(self.n_concepts, self.n_tasks, self.affine_hids, self.float_type)
        elif name == "TL":
            return TransLayer(self.n_concepts, self.n_tasks, self.affine_hids, self.layer_cfg, self.linear_rank, self.linear_hids, self.float_type)
            

class AutoReg(Module):
    def __init__(self, n_concepts, n_tasks, prior_units, prior_layers, prior_hids, n_components, float_type):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.prior_units = prior_units
        self.prior_layers = prior_layers
        self.prior_hids = prior_hids
        self.n_components = n_components
        self.rnn_cell = torch.nn.GRU(
            input_size = self.n_concepts * 3 + self.n_tasks + 1, 
            hidden_size = self.prior_units,
            num_layers = self.prior_layers,
            batch_first = True
        )

        rnn_out = []
        for i, h in enumerate(self.prior_hids):
            rnn_out.append(torch.nn.Linear(self.prior_units + self.n_concepts * 3 + self.n_tasks if i == 0 else self.prior_hids[i-1], h))
            rnn_out.append(torch.nn.Tanh())
        rnn_out.append(torch.nn.Linear(self.prior_hids[-1], self.n_components * 3))
        self.rnn_out = torch.nn.Sequential(*rnn_out)

        self.float_type = float_type
        
    def logp(self, z, c, b, m):
        B = z.shape[0]
        d = self.n_concepts
        state = torch.zeros(self.prior_layers, B, self.prior_units).to(c.device)
        z_t = -torch.ones((B,1), dtype = torch.float64 if self.float_type == "float64" else torch.float32).to(c.device)
        p_list = []
        for t in range(d):
            inp = torch.cat([z_t, c, b, m], dim = 1)
            inp = inp.unsqueeze(1)
            h_t, state = self.rnn_cell(inp, state)
            h_t = torch.squeeze(h_t, 1)
            h_t = torch.cat([h_t, c, b, m], dim = 1)
            p_t = self.rnn_out(h_t)
            p_list.append(p_t)
            z_t = torch.unsqueeze(z[:,t], dim = 1)
        params = torch.stack(p_list, dim = 1)
        log_like1_prev = mixture_likelihoods(params, z, self.n_components).to(c.device)
        query = m * (1 - b)
        mask, _ = torch.sort(query, dim = 1, descending = True, stable=True)
        log_likel = torch.sum(log_like1_prev * mask, dim = 1)

        return log_likel
        
    def sample(self, c, b, m):
        B = c.shape[0]
        d = self.n_concepts
        
        state = torch.zeros(self.prior_layers, B, self.prior_units).to(c.device)
        z_t = -torch.ones((B,1), dtype = torch.float64 if self.float_type == "float64" else torch.float32).to(c.device)
        z_list = []
        for t in range(d):
            inp = torch.cat([z_t, c, b, m], dim = 1)
            inp = inp.unsqueeze(1)
            h_t, state = self.rnn_cell(inp, state)
            h_t = torch.squeeze(h_t, 1)
            h_t = torch.cat([h_t, c, b, m], dim = 1)
            p_t = self.rnn_out(h_t)
            z_t = mixture_sample_dim(p_t, self.n_components).to(c.device)
            z_list.append(z_t)
        z = torch.concat(z_list, dim=1)
        return z

    def mean(self, c, b, m):
        B = c.shape[0]
        d = self.n_concepts
        
        state = torch.zeros(self.prior_layers, B, self.prior_units).to(c.device)
        z_t = -torch.ones((B,1), dtype = torch.float64 if self.float_type == "float64" else torch.float32).to(c.device)
        z_list = []
        for t in range(d):
            inp = torch.cat([z_t, c, b, m], dim = 1)
            inp = inp.unsqueeze(1)
            h_t, state = self.rnn_cell(inp, state)
            h_t = torch.squeeze(h_t, 1)
            h_t = torch.cat([h_t, c, b, m], dim = 1)
            p_t = self.rnn_out(h_t)
            z_t = mixture_mean_dim(p_t, self.n_components).to(c.device)
            z_list.append(z_t)
        z = torch.concat(z_list, dim=1)
        return z

def mixture_likelihoods(params, targets, n_components, base_distribution='gaussian'):
    '''
    Args:
        params: [B,d,c*3]
        targets: [B,d]
    Return:
        log_likelihood: [B,d]
    '''
    targets = torch.unsqueeze(targets, dim = -1)
    logits, means, lsigmas = torch.split(params, n_components, dim=2)
    sigmas = torch.exp(lsigmas)
    if base_distribution == 'gaussian':
        log_norm_consts = -lsigmas - 0.5 * np.log(2.0 * np.pi)
        log_kernel = -0.5 * torch.square((targets - means) / sigmas)
    elif base_distribution == 'laplace':
        log_norm_consts = -lsigmas - np.log(2.0)
        log_kernel = -torch.abs(targets - means) / sigmas
    elif base_distribution == 'logistic':
        log_norm_consts = -lsigmas
        diff = (targets - means) / sigmas
        log_kernel = -F.softplus(diff) - F.softplus(-diff)
    else:
        raise NotImplementedError
    log_exp_terms = log_kernel + log_norm_consts + logits
    log_likelihoods = torch.logsumexp(log_exp_terms, dim = -1) - torch.logsumexp(logits, dim = -1)
    
    return log_likelihoods

def mixture_sample_dim(params_dim, n_components, base_distribution='gaussian'):
    '''
    Args:
        params_dim: [B,n*3]
    Return:
        samp: [B,1]
    '''
    B = params_dim.shape[0]
    logits, means, lsigmas = torch.split(params_dim, n_components, dim=1)
    sigmas = torch.exp(lsigmas)
    # sample multinomial
    logits = torch.exp(logits)
    js = torch.multinomial(logits, 1)  # int64
    inds = torch.cat([torch.unsqueeze(torch.arange(B).to(js.device), dim = -1), js], dim = 1)
    # Sample from base distribution.
    if base_distribution == 'gaussian':
        zs = torch.normal(mean = torch.zeros((B, 1))).to(sigmas.device)
    elif base_distribution == 'laplace':
        zs = torch.log(torch.rand(size = (B, 1)).to(sigmas.device)) - \
            torch.log(torch.rand(size = (B, 1)).to(sigmas.device))
    elif base_distribution == 'logistic':
        x = torch.rand(size = (B, 1)).to(sigmas.device)
        zs = torch.log(x) - torch.log(1.0 - x)
    else:
        raise NotImplementedError()
    # scale and shift
    mu_zs = torch.unsqueeze(means[list(inds.T)], dim = -1)
    sigma_zs = torch.unsqueeze(sigmas[list(inds.T)], dim = -1)
    samp = sigma_zs * zs + mu_zs
    
    return samp

def mixture_mean_dim(params_dim, n_components, base_distribution='gaussian'):
    logits, means, lsigmas = torch.split(params_dim, n_components, dim=1)
    weights = torch.nn.Softmax(logits, dim=-1)

    return torch.sum(weights * means, dim=1, keepdims=True)

class ACFlowTransformDataset(Dataset):
    def __init__(self, dataset, n_tasks):
        self.dataset = dataset
        self.n_tasks = n_tasks

    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], list):
            y, c = batch[1]
        else:
            y, c = batch[1], batch[2]
        if len(batch) > 3:
            competencies = batch[3]
        else:
            competencies = None
        if len(batch) > 4:
            prev_interventions = batch[4]
        else:
            prev_interventions = None
        return x, y, (c, competencies, prev_interventions)
    
    def transform(self, batch):
        _, y, (x, _, _) = self._unpack_batch(batch)
        d = x.shape[-1]
        b = np.zeros([d], dtype=np.float32)
        no = np.random.choice(d+1)
        o = np.random.choice(d, [no], replace=False)
        b[o] = 1.
        m = b.copy()
        w = list(np.where(b == 0)[0])
        w.append(-1)
        w = np.random.choice(w)
        if w >= 0:
            m[w] = 1.
        b = torch.tensor(b)
        m = torch.tensor(m)
        y = y.to(torch.int64)
        return {'x': x, 'b': b, 'm': m, 'y': y}

    def transform_batch(x, y):
        B = x.shape[0]
        d = x.shape[-1]
        b = np.zeros([B, d], dtype=np.float32)
        m = np.zeros([B, d], dtype=np.float32)
        for i in range(B):            
            no = np.random.choice(d+1)
            o = np.random.choice(d, [no], replace=False)
            b[i][o] = 1.
            w = list(np.where(b[i] == 0)[0])
            w.append(-1)
            w = np.random.choice(w)
            if w >= 0:
                m[i][w] = 1.
        b = torch.tensor(b).to(x.device)
        m = torch.tensor(m).to(x.device)
        y = y.clone().to(torch.int64)
        return x, b, m, y

    def __getitem__(self, index):
        return self.transform(self.dataset[index])

    def __len__(self):
        return len(self.dataset)