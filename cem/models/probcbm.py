"""
Implementation of Kim et al.'s "Probabilistic Concept Bottleneck Models" as
introduced by their ICML 2023 paper (https://arxiv.org/abs/2306.01574).

This implementation is based on their official implementation found in their
accompying repository (https://github.com/ejkim47/prob-cbm) as of March 25th,
2024.
"""

import math
import numpy as np
import pytorch_lightning as pl
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18

from cem.metrics.accs import compute_accuracy
from cem.models.cbm import ConceptBottleneckModel
import cem.train.utils as utils




def weights_init(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
        stdv = 1. / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def sample_gaussian_tensors(mu, logsigma, num_samples):
    eps = torch.randn(mu.size(0), mu.size(1), num_samples, mu.size(2), dtype=mu.dtype, device=mu.device)
    samples_sigma = eps.mul(torch.exp(logsigma.unsqueeze(2) * 0.5))
    samples = samples_sigma.add_(mu.unsqueeze(2))
    return samples

def batchwise_cdist(samples1, samples2, eps=1e-6):
    """Compute L2 distance between each pair of the two multi-head embeddings in batch-wise.
    We may assume that samples have shape N x K x D, N: batch_size, K: number of embeddings, D: dimension of embeddings.
    The size of samples1 and samples2 (`N`) should be either
    - same (each sample-wise distance will be computed separately)
    - len(samples1) = 1 (samples1 will be broadcasted into samples2)
    - len(samples2) = 1 (samples2 will be broadcasted into samples1)
    The following broadcasting operation will be computed:
    (N x Nc x 1 x K x D) - (N x Nc x K x 1 x D) = (N x Nc x K x K x D)
    Parameters
    ----------
    samples1: torch.Tensor (shape: N x Nc x K x D)
    samples2: torch.Tensor (shape: N x Nc x K x D)
    Returns
    -------
    batchwise distance: N x Nc x K ** 2
    """
    if len(samples1.size()) not in [3, 4, 5] or len(samples2.size()) not in [3, 4, 5]:
        raise RuntimeError('expected: 4-dim tensors, got: {}, {}'.format(samples1.size(), samples2.size()))

    if samples1.size(0) == samples2.size(0):
        batch_size = samples1.size(0)
    elif samples1.size(0) == 1:
        batch_size = samples2.size(0)
    elif samples2.size(0) == 1:
        batch_size = samples1.size(0)
    elif samples1.shape[1] == samples2.shape[1]:
        samples1 = samples1.unsqueeze(2)
        samples2 = samples2.unsqueeze(3)
        samples1 = samples1.unsqueeze(1)
        samples2 = samples2.unsqueeze(0)
        result = torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps)
        return result.view(*result.shape[:-2], -1)
    else:
        raise RuntimeError(f'samples1 ({samples1.size()}) and samples2 ({samples2.size()}) dimensionalities '
                           'are non-broadcastable.')
    if len(samples1.size()) == 5:
        return torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps)
    elif len(samples1.size()) == 4:
        samples1 = samples1.unsqueeze(2)
        samples2 = samples2.unsqueeze(3)
        return torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps).view(batch_size, samples1.size(1), -1)
    else:
        samples1 = samples1.unsqueeze(1)
        samples2 = samples2.unsqueeze(2)
        return torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps).view(batch_size, -1)


def MC_dropout(act_vec, p=0.5, mask=True):
    return F.dropout(act_vec, p=p, training=mask)

class MultiHeadSelfAttention(nn.Module):
    """
    Self-attention module by Lin, Zhouhan, et al. ICLR 2017
    Code taken from Kim et al.'s https://github.com/ejkim47/prob-cbm/blob/main/models/module.py
    """

    def __init__(self, n_head, d_in, d_hidden):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        # This expects input x to be of size (b x seqlen x d_feat)
        attn = self.w_2(self.tanh(self.w_1(x)))
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1, 2, 0)
            attn.masked_fill_(mask, -np.inf)
        attn = self.softmax(attn)

        output = torch.bmm(attn.transpose(1, 2), x)
        if output.shape[1] == 1:
            output = output.squeeze(1)
        return output, attn


class PIENet(nn.Module):
    """
    Polysemous Instance Embedding (PIE) module
    Code taken from Kim et al.'s https://github.com/ejkim47/prob-cbm/blob/main/models/module.py
    """

    def __init__(self, n_embeds, d_in, d_out, d_h, dropout=0.0):
        super(PIENet, self).__init__()

        self.num_embeds = n_embeds
        self.f_fc = nn.Linear(d_in, d_out)
        self.attention = MultiHeadSelfAttention(n_embeds, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.f_fc.weight)
        nn.init.constant_(self.f_fc.bias, 0.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)
        residual = self.dropout(self.sigmoid(self.fc(residual)))
        out = self.f_fc(out)
        if self.num_embeds > 1:
            out = out.unsqueeze(1).repeat(1, self.num_embeds, 1)
        out = self.layer_norm(out + residual)
        return out, attn

class MCBCELoss(nn.Module):
    """
    Code adapted from Kim et al.'s https://github.com/ejkim47/prob-cbm/blob/main/utils/loss.py
    """
    def __init__(self, reduction='mean', criterion=None, weight=None, vib_beta=0.00005):
        super().__init__()
        if reduction not in {'mean', 'sum', None}:
            raise ValueError('unknown reduction {}'.format(reduction))
        self.reduction = reduction
        self.vib_beta = vib_beta
        self.criterion = (
            criterion or nn.BCELoss(reduction='none', weight=weight)
        )

    def kl_divergence(self, mu, logsigma, reduction='sum'):
        kl = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp())
        if reduction == 'sum':
            return kl.sum()
        else:
            return kl.sum(dim=-1).mean()

    def _compute_loss(self, probs, label):
        loss = self.criterion(probs, label)
        loss = loss.sum() if self.reduction == 'sum' else loss.mean()
        if loss != loss:
            print("NaN")
        return loss

    def forward(
        self,
        probs,
        image_mean,
        image_logsigma,
        concept_labels,
        negative_scale,
        **kwargs,
    ):
        vib_loss = 0
        loss_dict = {}

        if self.vib_beta != 0:
            vib_loss = self.kl_divergence(
                image_mean,
                image_logsigma,
                reduction=self.reduction,
            )
            loss_dict['vib_loss'] = vib_loss.item()

        t2i_loss = self._compute_loss(probs, concept_labels)
        loss = t2i_loss + self.vib_beta * vib_loss

        loss_dict['t2i_loss'] = t2i_loss.item()
        loss_dict['negative_scale'] = negative_scale.mean().item()
        loss_dict['loss'] = loss.mean().item()

        return loss, loss_dict

class UncertaintyModuleImage(nn.Module):
    """
    Code taken from Kim et al.'s https://github.com/ejkim47/prob-cbm/blob/main/models/module.py
    """
    def __init__(self, d_in, d_out, d_h):
        super().__init__()

        self.attention = MultiHeadSelfAttention(1, d_in, d_h)

        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

        self.fc2 = nn.Linear(d_in, d_out)
        self.embed_dim = d_in

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)

        fc_out = self.fc2(out)
        out = self.fc(residual) + fc_out

        return {
            'logsigma': out,
            'attention': attn,
        }

class ConceptConvModelBase(nn.Module):
    """
    Code taken from Kim et al.'s https://github.com/ejkim47/prob-cbm/blob/main/models/build_model_resnset.py
    """
    def __init__(
        self,
        c_extractor_arch=resnet18,
        pretrained=True,
        train_class_mode='sequential',
    ):
        nn.Module.__init__(self)
        self.train_class_mode = train_class_mode
        self.use_dropout = False

        base_model = c_extractor_arch(output_dim=1000)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.avgpool = base_model.avgpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.d_model = self.layer4[-1].conv2.out_channels

        self.cnn_module = nn.ModuleList([
            self.conv1,
            self.bn1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ])

    def forward_basic(self, x, avgpool=True, sample=False):
        if hasattr(self, 'features'):
            x = self.features(x)
            return x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_dropout:
            x = MC_dropout(x, p=0.2, mask=sample)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.use_dropout:
            x = MC_dropout(x, p=0.2, mask=sample)
        x = self.layer2(x)
        if self.use_dropout:
            x = MC_dropout(x, p=0.2, mask=sample)
        x = self.layer3(x)
        if self.use_dropout:
            x = MC_dropout(x, p=0.2, mask=sample)
        x = self.layer4(x)

        if avgpool:
            return self.avgpool(x)
        return x


class ProbConceptModel(ConceptConvModelBase):
    def __init__(
        self,
        n_concepts,
        c_extractor_arch=resnet18,
        pretrained=True,
        n_tasks=200,
        hidden_dim=16,
        n_samples_inference=50,
        use_neg_concept=True,
        pred_class=True,
        use_scale=True,
        activation_concept2class='prob',
        token2concept=None,
        train_class_mode='sequential',
        init_negative_scale=5,
        init_shift=5,
    ):
        """
        Code adapted from Kim et al.'s https://github.com/ejkim47/prob-cbm/blob/main/models/build_model_resnset.py
        """
        ConceptConvModelBase.__init__(
            self,
            c_extractor_arch=c_extractor_arch,
            pretrained=pretrained,
        )

        self.group2concept = token2concept
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.activation_concept2class = activation_concept2class
        self.train_class_mode = train_class_mode
        self.use_scale = use_scale

        self.mean_head = nn.Sequential(
            nn.Conv2d(self.d_model, n_concepts * hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(
                n_concepts * hidden_dim,
                n_concepts * hidden_dim,
                kernel_size=1,
                groups=n_concepts,
            ),
            nn.ReLU(),
            nn.Conv2d(
                n_concepts * hidden_dim,
                n_concepts * hidden_dim,
                kernel_size=1,
                groups=n_concepts,
            ),
        )
        self.logsigma_head = nn.Sequential(
            nn.Conv2d(self.d_model, n_concepts * hidden_dim, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(
                n_concepts * hidden_dim,
                n_concepts * hidden_dim,
                kernel_size=1,
                groups=n_concepts,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                n_concepts * hidden_dim,
                n_concepts * hidden_dim,
                kernel_size=1,
                groups=n_concepts,
            ),
        )

        weights_init(self.mean_head)
        weights_init(self.logsigma_head)

        self.use_neg_concept = use_neg_concept
        n_neg_concept = 1 if use_neg_concept else 0
        self.concept_vectors = nn.Parameter(
            torch.randn(n_neg_concept+1, n_concepts, hidden_dim),
            requires_grad=True,
        )
        negative_scale = init_negative_scale * torch.ones(1)
        shift = init_shift * torch.ones(1)
        nn.init.trunc_normal_(
            self.concept_vectors,
            std=1.0 / math.sqrt(hidden_dim),
        )

        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)

        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

        self.n_samples_inference = n_samples_inference

        self.pred_class = pred_class
        if pred_class:
            self.head = nn.Linear(n_concepts, n_tasks)
            if use_scale:
                scale = nn.Parameter(torch.ones(1) * 5, requires_grad=True)
                self.register_parameter('scale', scale)
            weights_init(self.head)

    def match_prob(
        self,
        sampled_image_features,
        sampled_attr_features,
        negative_scale=None,
        shift=None,
        reduction='mean',
    ):
        negative_scale = self.negative_scale if negative_scale is None else negative_scale
        shift = self.shift if shift is None else shift
        if not self.use_neg_concept:
            sampled_attr_features = (
                sampled_attr_features[1:]
                if sampled_attr_features.shape[0] > 1
                else sampled_attr_features
            )
            distance = batchwise_cdist(
                sampled_image_features,
                sampled_attr_features,
            )

            distance = distance.float()
            logits = (
                -negative_scale.view(1, -1, 1) * distance + shift.view(1, -1, 1)
            )
            prob = torch.exp(logits) / (torch.exp(logits) + torch.exp(-logits))

            if reduction == 'none':
                return logits, prob
            else:
                return logits.mean(axis=-1), prob.mean(axis=-1)
        else:
            distance = batchwise_cdist(
                sampled_image_features,
                sampled_attr_features,
            )
            distance = distance.permute(0, 2, 3, 1)

            logits = -self.negative_scale.view(1, -1, 1, 1) * distance
            prob = torch.nn.functional.softmax(logits, dim=-1)
            if reduction == 'none':
                return logits, prob
            return logits.mean(axis=-2), prob.mean(axis=-2)

    def get_uncertainty(self, pred_concept_logsigma):
        uncertainty = pred_concept_logsigma.mean(dim=-1).exp()
        return uncertainty

    def get_class_uncertainty(self, pred_concept_logsigma, weight):
        all_logsigma = pred_concept_logsigma.view(
            pred_concept_logsigma.shape[0],
            -1,
        )
        cov = all_logsigma
        cov = (
            torch.eye(cov.shape[1]).unsqueeze(0).to(cov.device) *
            cov.unsqueeze(-1)
        )
        full_cov = F.linear((F.linear(cov, weight)).transpose(1, 2), weight)
        _, s, _ = torch.linalg.svd(full_cov)
        C = (s + 1e-10).log()
        return C.mean(dim=1).exp()

    def get_uncertainty_with_matching_prob(
        self,
        sampled_embeddings,
        negative_scale=None,
    ):
        self_distance = torch.sqrt(
            (
                (
                    sampled_embeddings.unsqueeze(-2) -
                    sampled_embeddings.unsqueeze(-3)
                ) ** 2
            ).mean(-1) + 1e-10
        )
        eye = 1 - torch.eye(self_distance.size(-2)).view(-1)
        eye = eye.nonzero().contiguous().view(-1)
        logits = -self_distance.view(
            *sampled_embeddings.shape[:-2], -1
        )[..., eye] * (
            negative_scale if negative_scale is not None
            else 1
        )
        uncertainty = 1 - torch.sigmoid(logits).mean(dim=-1)
        return uncertainty

    def sample_embeddings(self, x, n_samples_inference=50):
        n_samples_inference = (
            self.n_samples_inference if n_samples_inference is None
            else n_samples_inference
        )
        B = x.shape[0]
        feature = self.forward_basic(x)
        pred_concept_mean = self.mean_head(feature).view(
            B,
            self.n_concepts,
            -1,
        )
        pred_concept_logsigma = self.logsigma_head(feature).view(
            B,
            self.n_concepts,
            -1,
        )
        pred_concept_mean = F.normalize(pred_concept_mean, p=2, dim=-1)

        pred_embeddings = sample_gaussian_tensors(
            pred_concept_mean,
            pred_concept_logsigma,
            self.n_samples_inference,
        )

        return {
            'pred_embeddings': pred_embeddings,
            'pred_mean': pred_concept_mean,
            'pred_logsigma': pred_concept_logsigma,
        }


    def forward(self, x, **kwargs):
        B = x.shape[0]
        feature = self.forward_basic(x)
        pred_concept_mean = self.mean_head(feature).view(
            B,
            self.n_concepts,
            -1,
        )
        pred_concept_logsigma = self.logsigma_head(feature).view(
            B,
            self.n_concepts,
            -1,
        )
        pred_concept_logsigma = torch.clip(pred_concept_logsigma, max=10)
        pred_concept_mean = F.normalize(pred_concept_mean, p=2, dim=-1)
        concept_mean = F.normalize(self.concept_vectors, p=2, dim=-1)

        if self.use_neg_concept and concept_mean.shape[0] < 2:
            concept_mean = torch.cat([-concept_mean, concept_mean])
            concept_logsigma = (
                torch.cat([concept_logsigma, concept_logsigma])
                if concept_logsigma is not None else None
            )

        pred_embeddings = sample_gaussian_tensors(
            pred_concept_mean,
            pred_concept_logsigma,
            self.n_samples_inference,
        ) # B x n_concepts x n_samples x hidden_dim
        concept_embeddings = concept_mean.unsqueeze(-2)

        concept_logit, concept_prob = self.match_prob(
            pred_embeddings,
            concept_embeddings,
        )
        concept_uncertainty = self.get_uncertainty(pred_concept_logsigma)

        if self.concept_pos_idx.sum() > 1:
            out_concept_prob, out_concept_idx = \
                concept_prob[:, self.concept_pos_idx==1].max(dim=1)
        else:
            out_concept_prob = (
                concept_prob[..., 1] if self.use_neg_concept
                else concept_prob
            )

        out_dict = {
            'pred_concept_prob': out_concept_prob,
            'pred_concept_uncertainty': concept_uncertainty,
            'pred_concept_logit': concept_logit,
            'pred_embeddings': pred_embeddings,
            'concept_embeddings': concept_embeddings,
            'pred_mean': pred_concept_mean,
            'pred_logsigma': pred_concept_logsigma,
            'concept_mean':concept_mean,
            'concept_logsigma': concept_logsigma,
            'shift': self.shift,
            'negative_scale': self.negative_scale,
            'concept_pos_idx': self.concept_pos_idx}

        if self.pred_class:
            out_concept_prob_d = out_concept_prob.detach()

            if hasattr(self, 'scale'):
                class_logits = self.head(out_concept_prob_d * self.scale.pow(2))
            else:
                class_logits = self.head(out_concept_prob_d)

            out_dict['pred_class_logit'] = class_logits

        return out_dict, {}

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'concept_vectors', 'class_vectors', 'shift', 'negative_scale'}



class ProbCBM(ProbConceptModel, ConceptBottleneckModel):
    """
    Code adapted from Kim et al.'s https://github.com/ejkim47/prob-cbm/blob/main/models/build_model_resnset.py
    """
    def __init__(
        self,
        n_concepts,
        n_tasks,
        concept_loss_weight=1,
        task_loss_weight=1,
        vib_beta=0.00005,
        warmup=False,

        hidden_dim=16,
        class_hidden_dim=128,
        intervention_prob=0.5,
        use_class_emb_from_concept=False,
        use_probabilistic_concept=True,

        c_extractor_arch=resnet18,
        pretrained=True,
        n_samples_inference=50,
        use_neg_concept=True,
        pred_class=True,
        use_scale=True,
        activation_concept2class='prob',
        token2concept=None,
        train_class_mode='sequential',
        init_negative_scale=5,
        init_shift=5,


        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        output_latent=False,
        use_concept_groups=False,

        top_k_accuracy=None,
    ):
        assert not output_latent, (
            f'Currently we have not yet added support for '
            'output_latent = False in ProbCBMs'
        )
        assert not output_interventions, (
            f'Currently we have not yet added support for '
            'output_interventions = False in ProbCBMs'
        )
        pl.LightningModule.__init__(self)
        ProbConceptModel.__init__(
            self,
            n_concepts=n_concepts,
            hidden_dim=hidden_dim,
            n_tasks=n_tasks,
            c_extractor_arch=c_extractor_arch,
            pretrained=pretrained,
            n_samples_inference=n_samples_inference,
            use_neg_concept=use_neg_concept,
            pred_class=pred_class,
            use_scale=use_scale,
            activation_concept2class=activation_concept2class,
            token2concept=token2concept,
            train_class_mode=train_class_mode,
            init_negative_scale=init_negative_scale,
            init_shift=init_shift,
        )
        self.train_class_mode = train_class_mode
        if train_class_mode in ['joint', 'independent']:
            self.stage = 'joint'
        elif train_class_mode == 'sequential':
            # Starts with concept training and then progresses to label
            # predictor training (i.e., stage = 'class')
            self.stage = 'concept'
        else:
            raise ValueError(
                f'Unsupported train_class_mode "{train_class_mode}"'
            )

        self.intervention_prob = intervention_prob
        self.use_class_emb_from_concept = use_class_emb_from_concept
        self.use_probabilistic_concept = use_probabilistic_concept
        self.warmup = warmup
        del self.mean_head
        del self.logsigma_head

        self.mean_head = nn.ModuleList([
            PIENet(1, self.d_model, hidden_dim, hidden_dim // 2)
            for _ in range(n_concepts)
        ])
        if self.use_probabilistic_concept:
            self.logsigma_head = nn.ModuleList([
                UncertaintyModuleImage(self.d_model, hidden_dim, hidden_dim // 2)
                for _ in range(n_concepts)
            ])

        if self.pred_class:
            del self.head

            class_hidden_dim = (
                class_hidden_dim if class_hidden_dim is not None
                else hidden_dim * 7
            )
            if not self.use_class_emb_from_concept:
                self.class_mean = nn.Parameter(
                    torch.randn(n_tasks, class_hidden_dim),
                    requires_grad=True,
                )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * n_concepts, class_hidden_dim)
            )

            if self.use_scale:
                self.class_negative_scale = nn.Parameter(
                    torch.ones(1) * 5,
                    requires_grad=True,
                )

            weights_init(self.head)

        del self.mean_head

        self.stem = nn.Sequential(
            nn.Conv2d(self.d_model, hidden_dim * n_concepts, kernel_size=1),
            nn.BatchNorm2d(hidden_dim * n_concepts),
            nn.ReLU(),
        )
        weights_init(self.stem)
        self.mean_head = nn.ModuleList([
            PIENet(1, hidden_dim, hidden_dim, hidden_dim)
            for _ in range(n_concepts)
        ])
        if self.use_probabilistic_concept:
            del self.logsigma_head
            self.logsigma_head = nn.ModuleList([
                UncertaintyModuleImage(hidden_dim, hidden_dim, hidden_dim)
                for _ in range(n_concepts)
            ])

        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.ones(n_concepts)

        self.loss_concept = MCBCELoss(weight=weight_loss, vib_beta=vib_beta)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                pos_weight=task_class_weights
            )
        )

        self.top_k_accuracy = top_k_accuracy
        self.concept_loss_weight = concept_loss_weight
        self.task_loss_weight = task_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.n_tasks = n_tasks
        self.intervention_policy = intervention_policy
        self.use_concept_groups = use_concept_groups
        self.output_interventions = output_interventions
        self.output_latent = output_latent

    def _train_step(
        self,
        stage,
        c,
        y,
        pred_concept_prob,
        pred_concept_logit,
        pred_class_logit,
        pred_class_prob,
        pred_mean,
        pred_logsigma,
        negative_scale,
        shift,
    ):
        if self.warmup:
            # Then we will freeze the weights of the CNN module as we are still
            # warming up
            for p in self.cnn_module.parameters():
                p.requires_grad = False
        else:
            for p in self.cnn_module.parameters():
                p.requires_grad = True
        loss, pred = 0, None
        loss_iter_dict = {}
        if isinstance(self.loss_concept, MCBCELoss):
            pred_concept = pred_concept_prob
            loss_concept, concept_loss_dict = self.loss_concept(
                probs=pred_concept_prob,
                image_mean=pred_mean,
                image_logsigma=pred_logsigma,
                concept_labels=c,
                negative_scale=negative_scale,
                shift=shift,
            )
            for k, v in concept_loss_dict.items():
                if k != 'loss':
                    loss_iter_dict['pcme_' + k] = v
        elif isinstance(self.loss_concept, nn.BCELoss):
            pred_concept = pred_concept_prob
            loss_concept = self.loss_concept(pred_concept, c)
        else:
            pred_concept = pred_concept_logit
            loss_concept = self.loss_concept(pred_concept, c)
            pred_concept = torch.sigmoid(pred_concept)

        if stage != 'class':
            loss += loss_concept * self.concept_loss_weight
        pred = pred_concept
        loss_iter_dict['concept'] = loss_concept

        if self.pred_class:
            if pred_class_logit is not None:
                pred_class = pred_class_logit
                loss_class = self.loss_task(pred_class, y)
                pred_class = F.softmax(pred_class, dim=-1)
            else:
                assert pred_class_prob is not None
                pred_class = pred_class_prob
                loss_class = F.nll_loss(pred_class.log(), y, reduction='mean')
            loss_iter_dict['class'] = loss_class

            if stage != 'concept':
                loss += loss_class * self.task_loss_weight
            pred = (
                pred_class if pred is None
                else torch.cat((pred_concept, pred_class), dim=1)
            )
        # And increase the number of epochs trained
        return loss, loss_iter_dict

    def _run_step(
        self,
        batch,
        batch_idx,
        train=False,
        intervention_idxs=None,
    ):
        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        c_sem, c_embs, y_probs, tail_outputs, _, _ = self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=train,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_embeddings=True,
            output_latent=True,
            output_interventions=False,
        )
        loss, loss_iter_dict = self._train_step(
            stage=self.stage,
            c=c,
            y=y,
            pred_concept_prob=c_sem,
            pred_concept_logit=tail_outputs.get('pred_concept_logit', None),
            pred_class_logit=tail_outputs.get('pred_class_logit', None),
            pred_class_prob=y_probs,
            pred_mean=tail_outputs.get('pred_mean', None),
            pred_logsigma=tail_outputs.get('pred_logsigma', None),
            negative_scale=tail_outputs.get('negative_scale', None),
            shift=tail_outputs.get('shift', None),
        )
        loss += self._extra_losses(
            x=x,
            y=y,
            c=c,
            c_sem=c_sem,
            c_pred=c_embs,
            y_pred=y_probs,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )
        # compute accuracy
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_sem,
            y_probs,
            c,
            y,
        )
        result = {
            "c_accuracy": c_accuracy,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "concept_loss": float(loss_iter_dict['concept'].detach().cpu().numpy()),
            "task_loss": float(loss_iter_dict['class'].detach().cpu().numpy()),
            "loss": float(loss.detach().cpu().numpy()) if not isinstance(loss, float) else loss,
            "avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
        }
        if self.top_k_accuracy is not None:
            y_true = y.reshape(-1).cpu().detach()
            y_pred = y_probs.cpu().detach()
            labels = list(range(self.n_tasks))
            for top_k_val in self.top_k_accuracy:
                y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                    y_true,
                    y_pred,
                    k=top_k_val,
                    labels=labels,
                )
                result[f'y_top_{top_k_val}_accuracy'] = y_top_k_accuracy
        return loss, result

    def _forward(
        self,
        x,
        intervention_idxs=None,
        c=None,
        y=None,
        train=False,
        latent=None,
        competencies=None,
        prev_interventions=None,
        output_embeddings=False,
        output_latent=None,
        output_interventions=False,
        inference_with_sampling=False,
        n_samples_inference=50,
    ):
        assert not output_interventions, (
            f'Currently we have not yet added support for '
            'output_interventions != None in ProbCBMs'
        )
        fwd_results = self.inner_forward(
            x=x,
            c=c,
            y=y,
            train=train,
            inference_with_sampling=inference_with_sampling,
            n_samples_inference=n_samples_inference,
        )
        c_sem = fwd_results.pop('pred_concept_prob')
        y_pred = fwd_results.pop('pred_class_prob')
        c_pred = fwd_results.pop('pred_embeddings')
        tail_results = []
        if output_interventions:
            raise NotImplementedError('output_interventions')
        if output_latent:
            tail_results.append(fwd_results)
        if output_embeddings:
            tail_results.append(self.concept_vectors[0, :, :].detach())
            tail_results.append(self.concept_vectors[1, :, :].detach())
        return tuple([c_sem, c_pred, y_pred] + tail_results)

    def predict_concept_dist(self, x):
        B = x.shape[0]
        feature = self.forward_basic(x, avgpool=False)
        feature = self.stem(feature)
        feature_avg = self.avgpool(feature).flatten(1)
        feature = feature.view(
            B,
            self.n_concepts,
            -1,
            feature.shape[-2:].numel(),
        ).transpose(2, 3)
        feature_avg = feature_avg.view(B, self.n_concepts, -1)
        pred_concept_mean = torch.stack(
            [
                mean_head(feature_avg[:, i], feature[:, i])[0]
                for i, mean_head in enumerate(self.mean_head)
            ],
            dim=1,
        )
        pred_concept_mean = F.normalize(pred_concept_mean, p=2, dim=-1)
        if self.use_probabilistic_concept:
            pred_concept_logsigma = torch.stack(
                [
                    logsigma_head(feature_avg[:, i], feature[:, i])['logsigma']
                    for i, logsigma_head in enumerate(self.logsigma_head)
                ],
                dim=1,
            )
            pred_concept_logsigma = torch.clip(pred_concept_logsigma, max=10)
            return pred_concept_mean, pred_concept_logsigma
        return pred_concept_mean, None


    def inner_forward(
        self,
        x,
        c=None,
        y=None,
        train=False,
        inference_with_sampling=False,
        n_samples_inference=50,
    ):
        pred_concept_mean, pred_concept_logsigma = self.predict_concept_dist(x)
        concept_mean = F.normalize(self.concept_vectors, p=2, dim=-1)

        if self.use_neg_concept and concept_mean.shape[0] < 2:
            concept_mean = torch.cat([-concept_mean, concept_mean])

        if self.use_probabilistic_concept:
            if train:
                pred_embeddings = sample_gaussian_tensors(
                    pred_concept_mean,
                    pred_concept_logsigma,
                    self.n_samples_inference,
                )
            else:
                if not inference_with_sampling:
                    pred_embeddings = pred_concept_mean.unsqueeze(2)
                else:
                    n_samples_inference = (
                        self.n_samples_inference if n_samples_inference is None
                        else n_samples_inference
                    )
                    pred_embeddings = sample_gaussian_tensors(
                        pred_concept_mean,
                        pred_concept_logsigma,
                        n_samples_inference,
                    )
        else:
            pred_embeddings = pred_concept_mean.unsqueeze(2)

        concept_embeddings = concept_mean.unsqueeze(-2)
        concept_logit, concept_prob = self.match_prob(
            pred_embeddings,
            concept_embeddings,
            reduction='none',
        )

        out_concept_prob = (
            concept_prob[..., 1].mean(dim=-1) if self.use_neg_concept
            else concept_prob.mean(dim=-1)
        )

        if self.use_probabilistic_concept:
            concept_uncertainty = self.get_uncertainty(pred_concept_logsigma)
            out_dict = {
                'pred_concept_prob': out_concept_prob,
                'pred_concept_uncertainty': concept_uncertainty,
                'pred_concept_logit': concept_logit,
                'pred_embeddings': pred_embeddings,
                'concept_embeddings': concept_embeddings,
                'pred_mean': pred_concept_mean,
                'pred_logsigma': pred_concept_logsigma,
                'concept_mean': concept_mean,
                'shift': self.shift,
                'negative_scale': self.negative_scale,
            }
        else:
            out_dict = {
                'pred_concept_prob': out_concept_prob,
                'pred_embeddings': pred_embeddings,
                'concept_embeddings': concept_embeddings,
                'pred_mean': pred_concept_mean,
                'concept_mean': concept_mean,
                'shift': self.shift,
                'negative_scale': self.negative_scale,
            }

        if self.pred_class:
            if self.train_class_mode in ['sequential', 'independent']:
                pred_embeddings_for_class = \
                    pred_embeddings.permute(0, 2, 1, 3).detach()
            elif self.train_class_mode == 'joint':
                pred_embeddings_for_class = pred_embeddings.permute(0, 2, 1, 3)
            else:
                raise NotImplementedError(
                    'train_class_mode should be one of [sequential, joint]'
                )

            if train:
                target_concept_onehot = F.one_hot(
                    c.long(),
                    num_classes=2,
                )
                gt_concept_embedddings = (
                    target_concept_onehot.view(*target_concept_onehot.shape, 1, 1) *
                    concept_embeddings.permute(1, 0, 2, 3).unsqueeze(0)
                ).sum(2)
                if self.train_class_mode == 'independent':
                    pred_embeddings_for_class = sample_gaussian_tensors(
                        gt_concept_embedddings.squeeze(-2),
                        pred_concept_logsigma.detach(),
                        self.n_samples_inference,
                    )
                    pred_embeddings_for_class = \
                        pred_embeddings_for_class.permute(0, 2, 1, 3).detach()
                if gt_concept_embedddings.shape[2] != self.n_samples_inference:
                    gt_concept_embedddings = gt_concept_embedddings.repeat(
                        1,
                        1,
                        self.n_samples_inference,
                        1,
                    )
                gt_concept_embedddings = \
                    gt_concept_embedddings.permute(0, 2, 1, 3).detach()
                pred_embeddings_for_class = torch.where(
                    torch.rand_like(gt_concept_embedddings[..., :1, :1]) < self.intervention_prob,
                    gt_concept_embedddings,
                    pred_embeddings_for_class,
                )

            pred_embeddings_for_class = pred_embeddings_for_class.contiguous().view(
                *pred_embeddings_for_class.shape[:2],
                -1,
            )
            pred_embeddings_for_class = self.head(pred_embeddings_for_class)
            out_dict['pred_embeddings_for_class'] = pred_embeddings_for_class
            class_mean = self.class_mean
            out_dict['class_mean'] = class_mean
            distance = torch.sqrt(
                (
                    (
                        pred_embeddings_for_class.unsqueeze(1) -
                        class_mean.unsqueeze(1).repeat(
                            1,
                            self.n_samples_inference if train else 1,
                            1,
                        ).unsqueeze(0)
                    ) ** 2
                ).mean(-1) + 1e-10
            )
            if self.use_scale:
                distance = self.class_negative_scale * distance
            out_class = F.softmax(-distance, dim=-2)
            out_dict['pred_class_prob'] = out_class.mean(dim=-1)
            if self.use_probabilistic_concept and not train:
                out_dict['pred_class_uncertainty'] = self.get_class_uncertainty(
                    pred_concept_logsigma.exp(),
                    self.head[0].weight,
                )

        return out_dict

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'concept_vectors',
            'class_vectors',
            'shift',
            'negative_scale',
            'class_mean',
        }

    @torch.jit.ignore
    def params_to_classify(self):
        output = []
        for n, _ in self.head.named_parameters():
            output.append('head.' + n)
        return output + ['class_mean', 'class_negative_scale']

    def predict_concept(self, x):
        pred_concept_mean, pred_concept_logsigma = self.predict_concept_dist(x)
        concept_mean = F.normalize(self.concept_vectors, p=2, dim=-1)

        if self.use_neg_concept and concept_mean.shape[0] < 2:
            concept_mean = torch.cat([-concept_mean, concept_mean])

        pred_embeddings = sample_gaussian_tensors(
            pred_concept_mean,
            pred_concept_logsigma,
            self.n_samples_inference,
        )
        concept_embeddings = concept_mean.unsqueeze(-2)

        concept_logit, concept_prob = self.match_prob(
            pred_embeddings,
            concept_embeddings,
            reduction='none',
        )
        concept_uncertainty = self.get_uncertainty(pred_concept_logsigma)

        out_concept_prob = (
            concept_prob[..., 1].mean(dim=-1) if self.use_neg_concept
            else concept_prob.mean(dim=-1)
        )

        return {
            'pred_concept_prob': out_concept_prob,
            'pred_concept_uncertainty': concept_uncertainty,
            'pred_embeddings': pred_embeddings,
            'pred_logsigma': pred_concept_logsigma,
        }

    def predict_class_with_gt_concepts(
        self,
        x,
        target_concept,
        order='uncertainty_avg',
        get_uncertainty=False,
    ):
        B = x.shape[0]
        out_dict = self.predict_concept(x)

        concept_mean = F.normalize(self.concept_vectors, p=2, dim=-1)
        concept_embeddings = concept_mean.unsqueeze(-2)

        target_concept_onehot = F.one_hot(target_concept.long(), num_classes=2)
        gt_concept_embedddings = (
            target_concept_onehot.view(*target_concept_onehot.shape, 1, 1) *
            concept_embeddings.permute(1, 0, 2, 3).unsqueeze(0)
        ).sum(2)

        pred_concept_embeddings = out_dict['pred_embeddings']
        n_groups = self.group2concept.shape[0]
        group2concept = self.group2concept.to(x.device)
        if order == 'uncertainty_avg':
            group_uncertainty = \
                (out_dict['pred_concept_uncertainty'] @ group2concept.t()) / group2concept.sum(dim=1).unsqueeze(0)
            _, intervention_order = group_uncertainty.sort(descending=True, dim=1)
        elif order == 'uncertainty_max':
            group_uncertainty, _ = (
                out_dict['pred_concept_uncertainty'].unsqueeze(1) *
                group2concept.unsqueeze(0)
            ).max(dim=-1)
            _, intervention_order = group_uncertainty.sort(descending=True, dim=1)
        else:
            assert isinstance(order, torch.Tensor)
            intervention_order = order.unsqueeze(0).repeat(B, 1)

        all_out_class = []
        all_uncertainty_class = []

        pred_concept_sigma = out_dict['pred_logsigma'].exp()
        for i in range(n_groups + 1):
            if i == 0:
                interventioned_concept_embedddings = pred_concept_embeddings
            else:
                inter_concepts_idx = group2concept[
                    intervention_order[:, :i]
                ].sum(dim=1)
                interventioned_concept_embedddings = torch.where(
                    inter_concepts_idx.view(*inter_concepts_idx.shape, 1, 1) == 1,
                    gt_concept_embedddings,
                    pred_concept_embeddings,
                )
                pred_concept_sigma = torch.where(
                    inter_concepts_idx.view(*inter_concepts_idx.shape, 1) == 1,
                    torch.zeros_like(pred_concept_sigma),
                    pred_concept_sigma,
                )
            interventioned_concept_embedddings = \
                interventioned_concept_embedddings.permute(
                    0,
                    2,
                    1,
                    3,
                ).contiguous().view(B, self.n_samples_inference, -1)
            pred_embeddings_for_class = self.head(
                interventioned_concept_embedddings
            )
            distance = torch.sqrt(
                (
                    (
                        pred_embeddings_for_class.unsqueeze(1) -
                        self.class_mean.unsqueeze(1).repeat(
                            1,
                            self.n_samples_inference,
                            1,
                        ).unsqueeze(0)
                    ) ** 2
                ).mean(-1) + 1e-10
            )
            out_class = F.softmax(-distance, dim=-2).mean(dim=-1)
            all_out_class.append(out_class)
            if get_uncertainty:
                all_uncertainty_class.append(
                    self.get_class_uncertainty(
                        pred_concept_sigma,
                        self.head[0].weight,
                    )
                )
        out_dict['all_pred_class_prob'] = all_out_class
        if get_uncertainty:
            out_dict['all_class_uncertainty'] = all_uncertainty_class

        return out_dict

