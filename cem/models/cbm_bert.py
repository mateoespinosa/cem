import numpy as np
import open_clip
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torchvision.models import resnet18


import cem.train.utils as utils
from cem.metrics.accs import compute_accuracy
from cem.models.cbm import ConceptBottleneckModel


def clip_preprocess_tensor(x):
    # Resize to CLIP’s expected 224x224 resolution
    x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)

    # Normalize to [0, 1] if needed
    if x.max() > 1.0:
        x = x / 255.0

    # Apply CLIP normalization
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device)[None, :, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device)[None, :, None, None]
    x = (x - mean) / std
    return x


class SoftCausalEncoderLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, attn_bias=None):
        # attn_bias: (S, S)
        q = k = v = src
        if attn_bias is not None:
            # Expand to match MultiheadAttention expectations: (num_heads * batch, S, S)
            bias = attn_bias.to(q.device)
            src2 = self.self_attn(q, k, v, attn_mask=bias)[0]
        else:
            src2 = self.self_attn(q, k, v, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# ============================================================
#  BertCBM — Concept-BERT with joint concept–label reasoning
# ============================================================

class BertCBM(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        concept_loss_weight=1,
        task_loss_weight=1,
        output_latent=False,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        lr_scheduler_factor=0.1,
        lr_scheduler_patience=10,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,

        top_k_accuracy=None,

        # Model-specific arguments
        concept_descriptions=None,
        d_model=512,
        n_heads=4,
        n_layers=2,
        mask_prob=0.5,
        task_mask_prob=None,
        train_steps=1,
        infer_steps=1,
        mask_id=-1,
        unmasked_loss_weight=0.1,

        c_extractor_arch=utils.wrap_pretrained_model(resnet18), #'clip',
        x2c_model=None,
        clip_weights=None, #"laion2b_s34b_b79k",
        pretrained_clip=None, #"ViT-B-32",
        freeze_backbone=False, #True,

        # NEW: concept head controls
        concept_head="linear", #"contrastive",  # "contrastive" or "linear"  # NEW
        concept_margin=2.0,          # scaling for +/- concept state init  # NEW
        normalize_concept_sims=True, # L2-normalize before sim             # NEW
        mask_rate=None,
        max_prob=1,
    ):
        """
        Concept-BERT with multi-step masked unmasking and y-as-token.
        The label y is treated as one of n_tasks + 1 embeddings
        (last = masked label token).
        """
        pl.LightningModule.__init__(self)

        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.output_latent = output_latent
        self.use_concept_groups = use_concept_groups
        self._intervention_idxs = None
        self.unmasked_loss_weight = unmasked_loss_weight
        self.mask_rate = mask_rate
        self.max_prob = max_prob
        self.start_rate = mask_prob

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
        self.concept_loss_weight = concept_loss_weight
        self.top_k_accuracy = top_k_accuracy

        # --- loss + optimizer hyperparams ---
        self.loss_concept = torch.nn.BCEWithLogitsLoss(
            weight=weight_loss,
            reduction='none',
        )
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights, reduction='none')
            if n_tasks > 1
            else torch.nn.BCEWithLogitsLoss(weight=task_class_weights, reduction='none',)
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.momentum = momentum

        # --- mask + iteration hyperparams ---
        self.mask_prob = mask_prob
        self.task_mask_prob = task_mask_prob or mask_prob
        self.train_steps = train_steps
        self.infer_steps = infer_steps
        self.mask_id = mask_id
        self._total_concept_loss = None
        self._total_task_loss = None

        # NEW: store head config
        self.concept_head = concept_head                      # NEW
        self.normalize_concept_sims = normalize_concept_sims  # NEW
        self.concept_margin = concept_margin                  # NEW

        # ------------------------------------------------------------
        # 1) CLIP visual backbone
        # ------------------------------------------------------------
        if (c_extractor_arch is None) or (
            isinstance(c_extractor_arch, str) and
            (c_extractor_arch.lower().strip() == "clip")
        ):
            clip_model, clip_preprocess, _ = open_clip.create_model_and_transforms(
                pretrained_clip,
                pretrained=clip_weights,
            )
            # to_pil = transforms.ToPILImage()
            # self.preprocess = lambda x: torch.stack([clip_preprocess(to_pil(img)).to(self.device) for img in x])
            self.preprocess = clip_preprocess_tensor
            self.visual_encoder = clip_model.visual

            encoder_dim = clip_model.text_projection.shape[1]
            self.using_clip = True
        else:
            self.preprocess = lambda x: x
            self.using_clip = False
            if x2c_model is not None:
                # Then this is assumed to be a module already provided as
                # the input to concepts method
                self.visual_encoder = x2c_model
            else:
                encoder_dim = 512
                self.visual_encoder = c_extractor_arch(
                    output_dim=encoder_dim
                )

        if freeze_backbone:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            self.visual_encoder.eval()

        # ------------------------------------------------------------
        # 2) Concept ID embeddings (+ optional CLIP text init)
        # ------------------------------------------------------------
        if (concept_descriptions is not None) and self.using_clip:
            clip_tokenizer = open_clip.get_tokenizer(pretrained_clip)
            with torch.no_grad():
                txt = clip_tokenizer(concept_descriptions)
                txt_emb = clip_model.encode_text(txt)
                txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            self.concept_id_emb = torch.nn.Embedding.from_pretrained(txt_emb, freeze=False)
        else:
            self.concept_id_emb = torch.nn.Embedding(
                self.n_concepts,
                encoder_dim,
            )
            torch.nn.init.normal_(self.concept_id_emb.weight, std=0.1)

        # Label embeddings: one per class + one for "masked" label
        self.class_emb = torch.nn.Embedding(n_tasks + 1, d_model)
        self.init_proj = torch.nn.Linear(encoder_dim, encoder_dim, bias=False)

        # ------------------------------------------------------------
        # 3) Transformer encoder backbone
        # ------------------------------------------------------------
        self.feature_proj = torch.nn.Linear(encoder_dim, d_model)
        self.state_emb = torch.nn.Embedding(3, d_model)  # 0=absent,1=present,2=mask

        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            4 * d_model,
            batch_first=True,
        )
        # enc_layer = SoftCausalEncoderLayer(
        #     d_model,
        #     n_heads,
        #     4 * d_model,
        #     batch_first=True,
        # )
        self.transformer = torch.nn.TransformerEncoder(enc_layer, n_layers)
        self.attn_bias = self._build_soft_causal_mask()


        self.pos_emb = torch.nn.Parameter(
            torch.randn(1, n_concepts + 2, d_model) * 0.02
        )

        # ------------------------------------------------------------
        # 4) Concept decoder
        # ------------------------------------------------------------
        # self.concept_decoder = torch.nn.Linear(d_model, 1)
        if self.concept_head == "contrastive":  # NEW
            self.concept_state_emb = torch.nn.Parameter(               # NEW
                torch.randn(n_concepts, 2, d_model) * 0.02            # NEW
            )                                                          # NEW
            torch.nn.init.xavier_uniform_(self.concept_state_emb)      # NEW
            # margin-separated init to break symmetry                  # NEW
            with torch.no_grad():                                      # NEW
                self.concept_state_emb[:, 1, :].mul_( self.concept_margin)  # pos  # NEW
                self.concept_state_emb[:, 0, :].mul_(-self.concept_margin)  # neg  # NEW
        else:  # "linear"                                              # NEW
            self.concept_decoder = torch.nn.Linear(d_model, 1)         # NEW
            self.c_temp = torch.nn.Parameter(torch.tensor(1.0))


        self.task_decoder = torch.nn.Linear(d_model, n_tasks)         # NEW

    # ------------------------------------------------------------
    # Mask utilities
    # ------------------------------------------------------------
    def mask_inputs(self, c_true, y_true=None, masked_c=None, c_mask=None):
        B, K = c_true.shape
        if c_mask is None:
            c_mask = torch.rand(B, K, device=self.device) < self.mask_prob
        if masked_c is None:
            masked_c = c_true.clone()
            with torch.no_grad():
                masked_c[c_mask] = self.mask_id
        if y_true is not None:
            y_mask = torch.rand(B, device=self.device) < self.task_mask_prob
        else:
            y_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
        return masked_c, c_mask, y_mask

    def _build_soft_causal_mask(
        self,
        concept_to_image_bias=2.0,
        concept_to_concept_bias=-1,
        task_to_image_bias=-float('inf'),
    ):
        """
        Builds a soft causal attention mask for [img | c_1..c_k | y].
        Negative numbers are additive biases to attention logits.
        """
        S = 1 + self.n_concepts + 1
        bias = torch.zeros(S, S, device=self.device)

        img_idx = 0
        c_start, c_end = 1, 1 + self.n_concepts
        y_idx = c_end

        # Concept tokens prefer image; weakly discourage other concepts
        bias[c_start:c_end, c_start:c_end] = concept_to_concept_bias
        bias[c_start:c_end, img_idx] = concept_to_image_bias
       # Fill diagonal manually (safe for older PyTorch versions)
        diag_idx = torch.arange(bias.size(0), device=bias.device)
        bias[diag_idx, diag_idx] = 0.0

        # Task token: attend to concepts only, not image
        bias[y_idx, img_idx] = task_to_image_bias
        bias[y_idx, c_start:c_end] = 0.0  # attend to concepts normally

        return bias.squeeze(0)

    def _build_attention_mask(self, seq_len, n_concepts, y_masked, device):
        S = 1 + self.n_concepts + 1
        attn_mask = torch.zeros(S, S, device=device)
        attn_mask[-1, 0] = float("-inf")  # task can't attend to image
        attn_mask[-1, -1] = 0             # can attend to itself
        # rest 0 => attends to all concepts
        return attn_mask

    # ------------------------------------------------------------
    # Token construction
    # ------------------------------------------------------------
    def _context_and_tokens(self, img_feat, c_vals, y=None, y_mask=None):
        B, K = c_vals.shape
        img_emb = self.feature_proj(img_feat).unsqueeze(1)

        concept_ids = torch.arange(K, device=img_feat.device).unsqueeze(0).expand(B, -1)
        id_emb = self.feature_proj(self.concept_id_emb(concept_ids))
        absent_emb = self.state_emb(torch.zeros_like(concept_ids))
        present_emb = self.state_emb(torch.ones_like(concept_ids))
        mask_emb = self.state_emb(torch.full_like(concept_ids, 2))
        c_vals_clamped = c_vals.clamp(-1, 1)
        token_emb = torch.where(
            (c_vals_clamped < 0).unsqueeze(-1),
            mask_emb,
            absent_emb * (1 - c_vals_clamped.unsqueeze(-1))
            + present_emb * c_vals_clamped.unsqueeze(-1),
        )
        token_emb = token_emb + id_emb

        # Per-sample y masking
        if y_mask is None:
            y_mask = torch.zeros(B, dtype=torch.bool, device=img_feat.device)
        masked_idx = torch.full(
            (B, 1),
            self.n_tasks,
            device=img_feat.device,
            dtype=torch.long,
        )
        if y is not None:
            y = y.view(-1, 1).clamp(0, self.n_tasks - 1).long()
            y_idx = torch.where(
                y_mask.view(-1, 1),
                masked_idx,
                y,
            )
        else:
            y_idx = masked_idx
        y_token_emb = self.class_emb(y_idx)
        tokens = torch.cat([img_emb, token_emb, y_token_emb], dim=1)

        # Normalize tokens
        tokens = F.layer_norm(tokens, tokens.shape[-1:])

        return tokens, img_feat

    def _propagate(
        self,
        img_feat,
        masked_c,
        y_mask=None,
        y=None,
    ):
        # Generate the tokenized seq [x_context | c_1 | c_2 | ... | c_k | y]
        tokens, _ = self._context_and_tokens(
            img_feat,
            c_vals=masked_c,
            y=y,
            y_mask=y_mask,
        )

        # # Generate the causal attention masks that ensure concepts attend
        # # x and y attends c and x.
        # attn_mask = self._build_attention_mask(
        #     seq_len=tokens.size(1),
        #     n_concepts=self.n_concepts,
        #     y_masked=y_mask,
        #     device=self.device,
        # )
        # h = self.transformer(tokens + self.pos_emb[:, :tokens.size(1), :], mask=attn_mask)

        # Pass the tokens through the self-attention transformer
        h = self.transformer(
            tokens + self.pos_emb[:, :tokens.size(1), :],
        )

        # h = tokens + self.pos_emb[:, :tokens.size(1), :]
        # for layer in self.transformer.layers:
        #     h = layer(h, attn_bias=self.attn_bias)


        # Pass the tokens through the self-attention transformer
        # attn_bias = self._build_soft_causal_mask()
        # h = self.transformer(tokens, mask=None, attn_bias=attn_bias)

        c_repr = h[:, 1:1 + self.n_concepts, :]
        y_repr = h[:, -1, :]

        c_repr = F.layer_norm(c_repr, c_repr.shape[-1:])


        # -----------------------------
        # Concept prediction head
        # -----------------------------
        if self.concept_head == "contrastive":
            # Concept logits via embedding similarity (contrastive)
            # c_repr: (B, K, d_model)
            # concept_state_emb: (K, 2, d_model)

            # Optional: L2-normalize before similarity for stability  # NEW
            if self.normalize_concept_sims:                             # NEW
                c_repr = F.normalize(c_repr, dim=-1)                    # NEW
                concept_pos = F.normalize(self.concept_state_emb[:, 1, :], dim=-1)  # NEW
                concept_neg = F.normalize(self.concept_state_emb[:, 0, :], dim=-1)  # NEW
            else:                                                       # NEW
                concept_pos = self.concept_state_emb[:, 1, :]           # NEW
                concept_neg = self.concept_state_emb[:, 0, :]           # NEW

            sim_pos = torch.sum(c_repr * concept_pos.unsqueeze(0), dim=-1)
            sim_neg = torch.sum(c_repr * concept_neg.unsqueeze(0), dim=-1)
            c_logits = torch.stack([sim_neg, sim_pos], dim=-1)
            c_sem = F.softmax(c_logits, dim=-1)[..., 1]
        else:
            # Classic sigmoid head                                         # NEW
            c_logits = self.concept_decoder(c_repr).squeeze(-1)            # NEW
            c_sem = torch.sigmoid(c_logits * self.c_temp)                                 # NEW

        # OLD contrastive:
        # # Similarly, compute the task probabilities using embedding
        # # similarities
        # E = self.class_emb.weight[:-1]
        # y_logits = torch.matmul(y_repr, E.T)
        # y_probs = F.softmax(y_logits, dim=-1)
        y_logits = self.task_decoder(y_repr)
        y_probs = F.softmax(y_logits, dim=-1)
        return h, c_logits, c_sem, y_logits, y_probs

    # def on_after_backward(self):
    #     # Sanity check: print gradient norms for key parameters
    #     for n, p in self.named_parameters():
    #         if p.grad is not None and "decoder" in n:
    #             print(f"[Grad check] {n}: {p.grad.norm().item():.6f}")

    # ------------------------------------------------------------
    # Forward (multi-step)
    # ------------------------------------------------------------
    def _forward(
        self,
        x,
        c=None,
        y=None,
        train=False,
        intervention_idxs=None,
        competencies=None,
        prev_interventions=None,
        latent=None,
        output_latent=None,
        output_embeddings=False,
        output_interventions=None,
        c_mask=None,
        masked_c=None,
    ):
        B = x.size(0)

        output_interventions = (
            output_interventions if output_interventions is not None
            else self.output_interventions
        )
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )
        B = x.size(0)

        # Next, set any intervened concepts to their provided values
        # For this, we will need a prior probability distribution as we will
        # need this for the intervention policy (if any)
        if latent is None:
            img_feat = self.visual_encoder(self.preprocess(x))
            if self.using_clip:
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            latent = img_feat
        else:
            img_feat = latent


        if train and (c is not None):
            # Then go ahead and randomly subsample the set of concepts
            # and tasks we will show as an input
            # print("self.mask_rate =", self.mask_rate)
            # print("self.max_prob =", self.max_prob)
            # print("self.mask_prob =", self.mask_prob)
            # print("self.task_mask_prob =", self.task_mask_prob)
            masked_c, c_mask, y_mask = self.mask_inputs(
                c,
                masked_c=masked_c,
                c_mask=c_mask,
                y_true=y,
            )
            if self.mask_rate not in [0, 1, None]:
                self.mask_prob = min(
                    self.max_prob,
                    self.start_rate * (self.mask_rate ** self.current_epoch)
                )
        else:
            # Else, our initial sample will be the same as the current_c
            # (which considers any interventions made on the concepts so
            # far)
            if masked_c is None:
                # We will start with all concepts being "masked"
                masked_c = torch.full(
                    (B, self.n_concepts),
                    self.mask_id,
                    device=x.device,
                )
            if c_mask is None:
                c_mask = (masked_c == self.mask_id)
            # And we will similarly mask the task label as we are about to
            # predict them. However, this masking will be done only if we have
            # not been provided with the task labels themselves already
            # FOR NOW (TODO: do the forward and backwards queries on a different
            # function as otherwise it is very hard to distinguish betwen
            # training-time and test-time instances where c and y are provided
            y_mask = torch.ones(B, dtype=torch.bool, device=x.device)
            # (
            #     torch.zeros(B, dtype=torch.bool, device=x.device)
            #     if (y is not None)
            #     else torch.ones(B, dtype=torch.bool, device=x.device)
            # )

        # And time to run the unmasking for a few steps (while accumulating
        # predictive losses)
        steps = self.train_steps if train else self.infer_steps
        total_c_loss, total_y_loss = 0.0, 0.0
        og_c_mask = c_mask.clone()
        for t in range(steps):
            h, c_logits, c_sem, y_logits, y_probs = self._propagate(
                img_feat=img_feat,
                masked_c=masked_c,
                y_mask=y_mask,
                y=y,
            )

            # At this point we can compute the task and concept losses
            # corresponding to this step
            if (c is not None):
                # Let's only consider the loss of the masked concepts
                total_c_loss += self.loss_concept(c_logits, c).mean()
                # all_c_losses = self.loss_concept(c_logits, c)
                # masked_loss = (
                #     all_c_losses * c_mask.type(all_c_losses.type())
                # )
                # masked_loss = masked_loss.mean()
                # if (~c_mask).any():
                #     unmasked_loss = (
                #         all_c_losses * (~c_mask).type(all_c_losses.type())
                #     ).mean()
                #     total_c_loss += masked_loss + self.unmasked_loss_weight * unmasked_loss
                # else:
                #     total_c_loss += masked_loss

            if (y is not None):
                total_y_loss += self.loss_task(y_logits, y).mean()
                # all_y_losses = self.loss_task(y_logits, y)
                # masked_loss = (
                #     all_y_losses * y_mask.type(all_y_losses.type())
                # )
                # # print("masked_loss.shape =", masked_loss.shape)
                # # print("masked_loss[0, :10] =", masked_loss[0, :10])
                # masked_loss = masked_loss.mean()
                # if (~y_mask).any():
                #     unmasked_loss = (
                #         all_y_losses * (~y_mask).type(all_y_losses.type())
                #     ).mean()
                #     total_y_loss += masked_loss + self.unmasked_loss_weight * unmasked_loss
                # else:
                #     total_y_loss += masked_loss

            # Finally, update the mask accordingly so that we are ready
            # for the next predictive step
            if t < steps - 1:
                with torch.no_grad():
                    conf = torch.abs(c_sem - 0.5)
                    if c_mask.any():
                        thr = conf[c_mask].quantile(0.5)
                        fill_mask = c_mask & (conf >= thr)
                        if train:
                            # out-of-place updates to avoid autograd in-place issues  # NEW
                            masked_c = torch.where(
                                fill_mask,
                                (c_sem > 0.5).type(masked_c.type()),
                                masked_c,
                            )
                            c_mask = torch.where(fill_mask, torch.zeros_like(c_mask), c_mask)  # NEW
                        else:
                            # also keep out-of-place for consistency                    # NEW
                            masked_c = torch.where(                                       # NEW
                                fill_mask,                                                # NEW
                                (c_sem > 0.5).long().type(masked_c.type()),               # NEW
                                masked_c,                                                 # NEW
                            )
                            c_mask = torch.where(fill_mask, torch.zeros_like(c_mask), c_mask)  # NEW

                    if y_mask.any():
                        maxp = y_probs.max(dim=1).values
                        thr_y = maxp[y_mask].quantile(0.5)
                        newly_fill = y_mask & (maxp >= thr_y)
                        # out-of-place update                                              # NEW
                        y_mask = torch.where(newly_fill, torch.zeros_like(y_mask), y_mask)     # NEW

        # Aggregate all the losses
        self._total_concept_loss = total_c_loss / max(steps, 1)
        self._total_task_loss = total_y_loss / max(steps, 1)
        # print("self._total_concept_loss =", self._total_concept_loss)
        # print("self._total_task_loss =", self._total_task_loss)

        # Time to perform an intervention
        if (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        ):
            pos_embeddings = torch.ones(c.shape, device=x.device).unsqueeze(-1)
            neg_embeddings = torch.zeros(c.shape, device=x.device).unsqueeze(-1)
            prior_distribution = self._prior_int_distribution(
                c=c,
                prob=c_sem,
                pos_embeddings=pos_embeddings,
                neg_embeddings=neg_embeddings,
                competencies=competencies,
                prev_interventions=prev_interventions,
                train=train,
                horizon=1,
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

            # Generate a new mask where we mask all unintervened concepts but use
            # the ground-truth values on the intervened concepts
            intervention_idxs = torch.tensor(intervention_idxs).to(c_int.device)
        else:
            c_int = c

        if (intervention_idxs is not None) and (c_int is not None):
            # And propagate again with the intervened concepts!
            masked_c = torch.where(
                intervention_idxs.bool(),
                c_int.type(masked_c.type()),
                (c_sem > 0.5).type(masked_c.type()),
            )
            h, c_logits, c_sem, y_logits, y_probs = self._propagate(
                img_feat=img_feat,
                masked_c=masked_c,
                y_mask=y_mask,
                y=y,
            )

        # # And make sure we set the probs of the concepts we were given
        # if c is not None:
        #     c_sem = torch.where(
        #         og_c_mask,
        #         c_sem,
        #         c
        #     )

        # And add any tail results we may want to add at the end
        tail_results = []
        if output_interventions:
            if isinstance(intervention_idxs, np.ndarray):
                intervention_idxs = torch.FloatTensor(intervention_idxs).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            tail_results.append(latent)
        tail_results += self._extra_tail_results(
            x=x,
            y=y,
            c=c,
            c_sem=c_sem,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )
        return tuple([c_sem, h, y_logits] + tail_results)

    # ------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------
    def _run_step(self, batch, batch_idx, train=False):
        x, y, (c, g, competencies, prev_interventions) = self._unpack_batch(
            batch
        )


        c_sem, bottleneck, y_logits = self._forward(
            x=x,
            c=c,
            y=y,
            train=train,
            prev_interventions=prev_interventions,
            competencies=competencies,
        )[:3]
        task_loss = self._total_task_loss
        concept_loss = self._total_concept_loss
        loss = self.concept_loss_weight * concept_loss + self.task_loss_weight * task_loss
        self._total_task_loss = self._total_concept_loss = None

        (c_acc, c_auc, c_f1), (y_acc, y_auc, y_f1) = compute_accuracy(
            c_sem,
            y_logits,
            c,
            y,
        )
        result = {
            "loss": loss.detach() if not isinstance(loss, (int, float)) else loss,
            "c_accuracy": c_acc,
            "y_accuracy": y_acc,
            "c_auc": c_auc,
            "y_auc": y_auc,
            "c_f1": c_f1,
            "y_f1": y_f1,
        }
        return loss, result