"""
Implementation of Kim et al.'s "Probabilistic Concept Bottleneck Models" as
introduced by their ICML 2023 paper (https://arxiv.org/abs/2306.01574).

This implementation is based on their official implementation found in their
accompying repository (https://github.com/ejkim47/prob-cbm) as of March 25th,
2024.
"""

import numpy as np
import pytorch_lightning as pl
import torch

from torchvision.models import resnet50

from cem.metrics.accs import compute_accuracy
from cem.models.cbm import ConceptBottleneckModel
import cem.train.utils as utils


from email.mime import base
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

from torchvision.models import resnet18


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
        backbone=resnet18,
        pretrained=True,
        train_class_mode='sequential',
    ):
        super(ConceptConvModelBase, self).__init__()
        self.train_class_mode = train_class_mode
        self.use_dropout = False

        base_model = backbone(pretrained=pretrained, num_classes=1000)
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
        num_concepts,
        backbone=resnet18,
        pretrained=True,
        num_classes=200,
        hidden_dim=128,
        n_samples_inference=7,
        use_neg_concept=False,
        pred_class=False,
        use_scale=False,
        activation_concept2class='prob',
        token2concept=None,
        train_class_mode='sequential',
        **kwargs,
    ):
        """
        Code taken from Kim et al.'s https://github.com/ejkim47/prob-cbm/blob/main/models/build_model_resnset.py
        """
        super(ProbConceptModel, self).__init__(
            backbone=backbone,
            pretrained=pretrained,
        )

        self.group2concept = token2concept
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.activation_concept2class = activation_concept2class
        self.train_class_mode = train_class_mode
        self.use_scale = use_scale

        self.mean_head = nn.Sequential(
            nn.Conv2d(self.d_model, num_concepts * hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(
                num_concepts * hidden_dim,
                num_concepts * hidden_dim,
                kernel_size=1,
                groups=num_concepts,
            ),
            nn.ReLU(),
            nn.Conv2d(
                num_concepts * hidden_dim,
                num_concepts * hidden_dim,
                kernel_size=1,
                groups=num_concepts,
            ),
        )
        self.logsigma_head = nn.Sequential(
            nn.Conv2d(self.d_model, num_concepts * hidden_dim, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(
                num_concepts * hidden_dim,
                num_concepts * hidden_dim,
                kernel_size=1,
                groups=num_concepts,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                num_concepts * hidden_dim,
                num_concepts * hidden_dim,
                kernel_size=1,
                groups=num_concepts,
            ),
        )

        weights_init(self.mean_head)
        weights_init(self.logsigma_head)

        self.use_neg_concept = use_neg_concept
        n_neg_concept = 1 if use_neg_concept else 0
        self.concept_vectors = nn.Parameter(
            torch.randn(n_neg_concept+1, num_concepts, hidden_dim),
            requires_grad=True,
        )
        negative_scale = kwargs.get('init_negative_scale', 1) * torch.ones(1)
        shift = kwargs.get('init_shift', 0) * torch.ones(1)
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
            self.head = nn.Linear(num_concepts, num_classes)
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

    def sample_embeddings(self, x, n_samples_inference=None):
        n_samples_inference = (
            self.n_samples_inference if n_samples_inference is None
            else n_samples_inference
        )
        B = x.shape[0]
        feature = self.forward_basic(x)
        pred_concept_mean = self.mean_head(feature).view(
            B,
            self.num_concepts,
            -1,
        )
        pred_concept_logsigma = self.logsigma_head(feature).view(
            B,
            self.num_concepts,
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
            self.num_concepts,
            -1,
        )
        pred_concept_logsigma = self.logsigma_head(feature).view(
            B,
            self.num_concepts,
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
        ) # B x num_concepts x n_samples x hidden_dim
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



class ProbCBM(ProbConceptModel):
    def __init__(
        self,
        num_concepts,
        hidden_dim=128,
        num_classes=200,
        class_hidden_dim=None,
        intervention_prob=False,
        use_class_emb_from_concept=False,
        use_probabilsitic_concept=True,
        **kwargs,
    ):
        super(ProbCBM, self).__init__(
            num_concepts=num_concepts,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            **kwargs,
        )

        self.intervention_prob = intervention_prob
        self.use_class_emb_from_concept = use_class_emb_from_concept
        self.use_probabilsitic_concept = use_probabilsitic_concept
        del self.mean_head
        del self.logsigma_head

        self.mean_head = nn.ModuleList([
            PIENet(1, self.d_model, hidden_dim, hidden_dim // 2)
            for _ in range(num_concepts)
        ])
        if self.use_probabilsitic_concept:
            self.logsigma_head = nn.ModuleList([
                UncertaintyModuleImage(self.d_model, hidden_dim, hidden_dim // 2)
                for _ in range(num_concepts)
            ])

        if self.pred_class:
            del self.head

            class_hidden_dim = (
                class_hidden_dim if class_hidden_dim is not None
                else hidden_dim * 7
            )
            if not self.use_class_emb_from_concept:
                self.class_mean = nn.Parameter(
                    torch.randn(num_classes, class_hidden_dim),
                    requires_grad=True,
                )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * num_concepts, class_hidden_dim)
            )

            if self.use_scale:
                self.class_negative_scale = nn.Parameter(
                    torch.ones(1) * 5,
                    requires_grad=True,
                )

            weights_init(self.head)

        del self.mean_head

        self.stem = nn.Sequential(
            nn.Conv2d(self.d_model, hidden_dim * num_concepts, kernel_size=1),
            nn.BatchNorm2d(hidden_dim * num_concepts),
            nn.ReLU(),
        )
        weights_init(self.stem)
        self.mean_head = nn.ModuleList([
            PIENet(1, hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_concepts)
        ])
        if self.use_probabilsitic_concept:
            del self.logsigma_head
            self.logsigma_head = nn.ModuleList([
                UncertaintyModuleImage(hidden_dim, hidden_dim, hidden_dim)
                for _ in range(num_concepts)
            ])


    def predict_concept_dist(self, x):
        B = x.shape[0]
        feature = self.forward_basic(x, avgpool=False)
        feature = self.stem(feature)
        feature_avg = self.avgpool(feature).flatten(1)
        feature = feature.view(
            B,
            self.num_concepts,
            -1,
            feature.shape[-2:].numel(),
        ).transpose(2, 3)
        feature_avg = feature_avg.view(B, self.num_concepts, -1)
        pred_concept_mean = torch.stack(
            [
                mean_head(feature_avg[:, i], feature[:, i])[0]
                for i, mean_head in enumerate(self.mean_head)
            ],
            dim=1,
        )
        pred_concept_mean = F.normalize(pred_concept_mean, p=2, dim=-1)
        if self.use_probabilsitic_concept:
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

    def forward(
        self,
        x,
        inference_with_sampling=False,
        n_samples_inference=None,
        **kwargs,
    ):
        B = x.shape[0]
        pred_concept_mean, pred_concept_logsigma = self.predict_concept_dist(x)
        concept_mean = F.normalize(self.concept_vectors, p=2, dim=-1)

        if self.use_neg_concept and concept_mean.shape[0] < 2:
            concept_mean = torch.cat([-concept_mean, concept_mean])

        if self.use_probabilsitic_concept:
            if self.training:
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

        if self.use_probabilsitic_concept:
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

            if self.training:
                target_concept_onehot = F.one_hot(
                    kwargs['target_concept'].long(),
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
                            self.n_samples_inference if self.training else 1,
                            1,
                        ).unsqueeze(0)
                    ) ** 2
                ).mean(-1) + 1e-10
            )
            if self.use_scale:
                distance = self.class_negative_scale * distance
            out_class = F.softmax(-distance, dim=-2)
            out_dict['pred_class_prob'] = out_class.mean(dim=-1)
            if self.use_probabilsitic_concept and not self.training:
                out_dict['pred_class_uncertainty'] = self.get_class_uncertainty(
                    pred_concept_logsigma.exp(),
                    self.head[0].weight,
                )

        return out_dict, {}

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


################################################################################
## OUR MODEL
################################################################################


class ProbCBM(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        training_intervention_prob=0.25,
        embedding_activation="leakyrelu",
        shared_prob_gen=True,
        concept_loss_weight=1,
        task_loss_weight=1,

        c2y_model=None,
        c2y_layers=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        output_latent=False,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,
        tau=1,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,

        top_k_accuracy=None,
    ):
        """
        Constructs a Concept Embedding Model (CEM) as defined by
        Espinosa Zarlenga et al. 2022.

        :param int n_concepts: The number of concepts given at training time.
        :param int n_tasks: The number of output classes of the CEM.
        :param int emb_size: The size of each concept embedding. Defaults to 16.
        :param float training_intervention_prob: RandInt probability. Defaults
            to 0.25.
        :param str embedding_activation: A valid nonlinearity name to use for the
            generated embeddings. It must be one of [None, "sigmoid", "relu",
            "leakyrelu"] and defaults to "leakyrelu".
        :param Bool shared_prob_gen: Whether or not weights are shared across
            all probability generators. Defaults to True.
        :param float concept_loss_weight: Weight to be used for the final loss'
            component corresponding to the concept classification loss. Default
            is 0.01.
        :param float task_loss_weight: Weight to be used for the final loss'
            component corresponding to the output task classification loss.
            Default is 1.

        :param Pytorch.Module c2y_model:  A valid pytorch Module used to map the
            CEM's bottleneck (with size n_concepts * emb_size) to `n_tasks`
            output activations (i.e., the output of the CEM).
            If not given, then a simple leaky-ReLU MLP, whose hidden
            layers have sizes `c2y_layers`, will be used.
        :param List[int] c2y_layers: List of integers defining the size of the
            hidden layers to be used in the MLP to predict classes from the
            bottleneck if c2y_model was NOT provided. If not given, then we will
            use a simple linear layer to map the bottleneck to the output classes.
        :param Fun[(int), Pytorch.Module] c_extractor_arch: A generator function
            for the latent code generator model that takes as an input the size
            of the latent code before the concept embedding generators act (
            using an argument called `output_dim`) and returns a valid Pytorch
            Module that maps this CEM's inputs to the latent space of the
            requested size.

        :param str optimizer:  The name of the optimizer to use. Must be one of
            `adam` or `sgd`. Default is `adam`.
        :param float momentum: Momentum used for optimization. Default is 0.9.
        :param float learning_rate:  Learning rate used for optimization.
            Default is 0.01.
        :param float weight_decay: The weight decay factor used during
            optimization. Default is 4e-05.
        :param List[float] weight_loss: Either None or a list with n_concepts
            elements indicating the weights assigned to each predicted concept
            during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.
        :param List[float] task_class_weights: Either None or a list with
            n_tasks elements indicating the weights assigned to each output
            class during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.

        :param List[float] active_intervention_values: A list of n_concepts
            values to use when positively intervening in a given concept (i.e.,
            setting concept c_i to 1 would imply setting its corresponding
            predicted concept to active_intervention_values[i]). If not given,
            then we will assume that we use `1` for all concepts. This
            parameter is important when intervening in CEMs that do not have
            sigmoidal concepts, as the intervention thresholds must then be
            inferred from their empirical training distribution.
        :param List[float] inactive_intervention_values: A list of n_concepts
            values to use when negatively intervening in a given concept (i.e.,
            setting concept c_i to 0 would imply setting its corresponding
            predicted concept to inactive_intervention_values[i]). If not given,
            then we will assume that we use `0` for all concepts.
        :param Callable[(np.ndarray, np.ndarray, np.ndarray), np.ndarray] intervention_policy:
            An optional intervention policy to be used when intervening on a
            test batch sample x (first argument), with corresponding true
            concepts c (second argument), and true labels y (third argument).
            The policy must produce as an output a list of concept indices to
            intervene (in batch form) or a batch of binary masks indicating
            which concepts we will intervene on.

        :param List[int] top_k_accuracy: List of top k values to report accuracy
            for during training/testing when the number of tasks is high.
        """
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.pre_concept_model = c_extractor_arch(output_dim=None)
        self.training_intervention_prob = training_intervention_prob
        self.output_latent = output_latent
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

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
        self.task_loss_weight = task_loss_weight
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        self.shared_prob_gen = shared_prob_gen
        self.top_k_accuracy = top_k_accuracy
        for i in range(n_concepts):
            if embedding_activation is None:
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            2 * emb_size,
                        ),
                    ])
                )
            elif embedding_activation == "sigmoid":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            2 * emb_size,
                        ),
                        torch.nn.Sigmoid(),
                    ])
                )
            elif embedding_activation == "leakyrelu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            2 * emb_size,
                        ),
                        torch.nn.LeakyReLU(),
                    ])
                )
            elif embedding_activation == "relu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            2 * emb_size,
                        ),
                        torch.nn.ReLU(),
                    ])
                )
            if self.shared_prob_gen and (
                len(self.concept_prob_generators) == 0
            ):
                # Then we will use one and only one probability generator which
                # will be shared among all concepts. This will force concept
                # embedding vectors to be pushed into the same latent space
                self.concept_prob_generators.append(torch.nn.Linear(
                    2 * emb_size,
                    1,
                ))
            elif not self.shared_prob_gen:
                self.concept_prob_generators.append(torch.nn.Linear(
                    2 * emb_size,
                    1,
                ))
        if c2y_model is None:
            # Else we construct it here directly
            units = [
                n_concepts * emb_size
            ] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model
        self.sig = torch.nn.Sigmoid()

        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights
            )
        )
        self.concept_loss_weight = concept_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.n_tasks = n_tasks
        self.emb_size = emb_size
        self.tau = tau
        self.use_concept_groups = use_concept_groups


    def _after_interventions(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        intervention_idxs=None,
        c_true=None,
        train=False,
        competencies=None,
    ):
        if train and (self.training_intervention_prob != 0) and (
            (c_true is not None) and
            (intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(
                self.ones * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if (c_true is None) or (intervention_idxs is None):
            return prob, intervention_idxs
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        return prob * (1 - intervention_idxs) + intervention_idxs * c_true, intervention_idxs

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
        output_interventions=None,
    ):
        output_interventions = (
            output_interventions if output_interventions is not None
            else self.output_interventions
        )
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )
        if latent is None:
            pre_c = self.pre_concept_model(x)
            contexts = []
            c_sem = []

            # First predict all the concept probabilities
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context = context_gen(pre_c)
                prob = prob_gen(context)
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(self.sig(prob))
            c_sem = torch.cat(c_sem, axis=-1)
            contexts = torch.cat(contexts, axis=1)
            latent = contexts, c_sem
        else:
            contexts, c_sem = latent

        # Now include any interventions that we may want to perform!
        if (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        ):
            horizon = self.intervention_policy.num_groups_intervened
            if hasattr(self.intervention_policy, "horizon"):
                horizon = self.intervention_policy.horizon
            prior_distribution = self._prior_int_distribution(
                prob=c_sem,
                pos_embeddings=contexts[:, :, :self.emb_size],
                neg_embeddings=contexts[:, :, self.emb_size:],
                competencies=competencies,
                prev_interventions=prev_interventions,
                c=c,
                train=train,
                horizon=horizon,
            )
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=prior_distribution,
            )

        else:
            c_int = c
        if not train:
            intervention_idxs = self._standardize_indices(
                intervention_idxs=intervention_idxs,
                batch_size=x.shape[0],
            )

        # Then, time to do the mixing between the positive and the
        # negative embeddings
        probs, intervention_idxs = self._after_interventions(
            c_sem,
            pos_embeddings=contexts[:, :, :self.emb_size],
            neg_embeddings=contexts[:, :, self.emb_size:],
            intervention_idxs=intervention_idxs,
            c_true=c_int,
            train=train,
            competencies=competencies,
        )
        # Then time to mix!
        c_pred = (
            contexts[:, :, :self.emb_size] * torch.unsqueeze(probs, dim=-1) +
            contexts[:, :, self.emb_size:] * (1 - torch.unsqueeze(probs, dim=-1))
        )
        c_pred = c_pred.view((-1, self.emb_size * self.n_concepts))
        y = self.c2y_model(c_pred)
        tail_results = []
        if output_interventions:
            if (
                (intervention_idxs is not None) and
                isinstance(intervention_idxs, np.ndarray)
            ):
                intervention_idxs = torch.FloatTensor(
                    intervention_idxs
                ).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            tail_results.append(latent)
        if output_embeddings:
            tail_results.append(contexts[:, :, :self.emb_size])
            tail_results.append(contexts[:, :, self.emb_size:])
        return tuple([c_sem, c_pred, y] + tail_results)
