import math

import torch
from torch.distributions import kl_divergence
from torch.nn import Module

class VAEAC(Module):
    """
    Variational Autoencoder with Arbitrary Conditioning core model.
    It is rather flexible, but have several assumptions:
    + The batch of objects and the mask of unobserved features
      have the same shape.
    + The prior and proposal distributions in the latent space
      are component-wise independent Gaussians.
    The constructor takes
    + Prior and proposal network which take as an input the concatenation
      of the batch of objects and the mask of unobserved features
      and return the parameters of Gaussians in the latent space.
      The range of neural networks outputs should not be restricted.
    + Generative network takes latent representation as an input
      and returns the parameters of generative distribution
      p_theta(x_b | z, x_{1 - b}, b), where b is the mask
      of unobserved features. The information about x_{1 - b} and b
      can be transmitted to generative network from prior network
      through nn_utils.MemoryLayer. It is guaranteed that for every batch
      prior network is always executed before generative network.
    + Reconstruction log probability. rec_log_prob is a callable
      which takes (groundtruth, distr_params, mask) as an input
      and return vector of differentiable log probabilities
      p_theta(x_b | z, x_{1 - b}, b) for each object of the batch.
    + Sigma_mu and sigma_sigma are the coefficient of the regularization
      in the hidden space. The default values correspond to a very weak,
      almost disappearing regularization, which is suitable for all
      experimental setups the model was tested on.
    """
    def __init__(self, rec_log_prob, proposal_network, prior_network,
                 generative_network, sigma_mu=1e4, sigma_sigma=1e-4):
        super().__init__()
        self.rec_log_prob = rec_log_prob
        self.proposal_network = proposal_network
        self.prior_network = prior_network
        self.generative_network = generative_network
        self.sigma_mu = sigma_mu
        self.sigma_sigma = sigma_sigma

    def _normal_parse_params(params, min_sigma=0):
        """
        Take a Tensor (e. g. neural network output) and return
        torch.distributions.Normal distribution.
        This Normal distribution is component-wise independent,
        and its dimensionality depends on the input shape.
        First half of channels is mean of the distribution,
        the softplus of the second half is std (sigma), so there is
        no restrictions on the input tensor.

        min_sigma is the minimal value of sigma. I. e. if the above
        softplus is less than min_sigma, then sigma is clipped
        from below with value min_sigma. This regularization
        is required for the numerical stability and may be considered
        as a neural network architecture choice without any change
        to the probabilistic model.
        """
        n = params.shape[0]
        d = params.shape[1]
        mu = params[:, :d // 2]
        sigma_params = params[:, d // 2:]
        sigma = softplus(sigma_params)
        sigma = sigma.clamp(min=min_sigma)
        distr = Normal(mu, sigma)
        return distr


    def _make_observed(self, batch, mask):
        """
        Copy batch of objects and zero unobserved features.
        """
        observed = torch.tensor(batch)
        observed[mask.byte()] = 0
        return observed

    def _make_latent_distributions(self, x, mask, y, no_proposal=False):
        """
        Make latent distributions for the given batch and mask.
        No no_proposal is True, return None instead of proposal distribution.
        """
        observed = self.make_observed(x, mask)
        if no_proposal:
            proposal = None
        else:
            full_info = torch.cat([x, mask], 1)
            proposal_params = self.proposal_network(full_info)
            proposal = self._normal_parse_params(proposal_params, 1e-3)
        prior_params = self.prior_network(torch.cat([x, mask], 1))
        prior = self._normal_parse_params(prior_params, 1e-3)
        return proposal, prior

    def _prior_regularization(self, prior):
        """
        The prior distribution regularization in the latent space.
        Though it saves prior distribution parameters from going to infinity,
        the model usually doesn't diverge even without this regularization.
        It almost doesn't affect learning process near zero with default
        regularization parameters which are recommended to be used.
        """
        num_objects = prior.mean.shape[0]
        mu = prior.mean.view(num_objects, -1)
        sigma = prior.scale.view(num_objects, -1)
        mu_regularizer = -(mu ** 2).sum(-1) / 2 / (self.sigma_mu ** 2)
        sigma_regularizer = (sigma.log() - sigma).sum(-1) * self.sigma_sigma
        return mu_regularizer + sigma_regularizer

    def batch_vlb(self, batch, mask):
        """
        Compute differentiable lower bound for the given batch of objects
        and mask.
        """
        x, y = batch
        proposal, prior = self._make_latent_distributions(x, mask)
        prior_regularization = self._prior_regularization(prior)
        latent = proposal.rsample()
        rec_params = self.generative_network(latent)
        rec_loss = self.rec_log_prob(batch, rec_params, mask)
        kl = kl_divergence(proposal, prior).view(batch.shape[0], -1).sum(-1)
        return rec_loss - kl + prior_regularization

    def batch_iwae(self, batch, mask, K):
        """
        Compute IWAE log likelihood estimate with K samples per object.
        Technically, it is differentiable, but it is recommended to use it
        for evaluation purposes inside torch.no_grad in order to save memory.
        With torch.no_grad the method almost doesn't require extra memory
        for very large K.
        The method makes K independent passes through generator network,
        so the batch size is the same as for training with batch_vlb.
        """
        proposal, prior = self._make_latent_distributions(batch, mask)
        estimates = []
        for i in range(K):
            latent = proposal.rsample()

            rec_params = self.generative_network(latent)
            rec_loss = self.rec_log_prob(batch, rec_params, mask)

            prior_log_prob = prior.log_prob(latent)
            prior_log_prob = prior_log_prob.view(batch.shape[0], -1)
            prior_log_prob = prior_log_prob.sum(-1)

            proposal_log_prob = proposal.log_prob(latent)
            proposal_log_prob = proposal_log_prob.view(batch.shape[0], -1)
            proposal_log_prob = proposal_log_prob.sum(-1)

            estimate = rec_loss + prior_log_prob - proposal_log_prob
            estimates.append(estimate[:, None])

        return torch.logsumexp(torch.cat(estimates, 1), 1) - math.log(K)

    def generate_samples_params(self, batch, mask, K=1):
        """
        Generate parameters of generative distributions for samples
        from the given batch.
        It makes K latent representation for each object from the batch
        and generate samples from them.
        The second axis is used to index samples for an object, i. e.
        if the batch shape is [n x D1 x D2], then the result shape is
        [n x K x D1 x D2].
        It is better to use it inside torch.no_grad in order to save memory.
        With torch.no_grad the method doesn't require extra memory
        except the memory for the result.
        """
        _, prior = self._make_latent_distributions(batch, mask)
        samples_params = []
        for i in range(K):
            latent = prior.rsample()
            sample_params = self.generative_network(latent)
            samples_params.append(sample_params.unsqueeze(1))
        return torch.cat(samples_params, 1)

    def generate_reconstructions_params(self, batch, mask, K=1):
        """
        Generate parameters of generative distributions for reconstructions
        from the given batch.
        It makes K latent representation for each object from the batch
        and generate samples from them.
        The second axis is used to index samples for an object, i. e.
        if the batch shape is [n x D1 x D2], then the result shape is
        [n x K x D1 x D2].
        It is better to use it inside torch.no_grad in order to save memory.
        With torch.no_grad the method doesn't require extra memory
        except the memory for the result.
        """
        _, prior = self._make_latent_distributions(batch, mask)
        reconstructions_params = []
        for i in range(K):
            latent = prior.rsample()
            rec_params = self.generative_network(latent)
            reconstructions_params.append(rec_params.unsqueeze(1))
        return torch.cat(reconstructions_params, 1)
