from typing import List

import torch
import torch.nn as nn

from torchattacks.attack import Attack

from attack.mmd import MMDLoss


class SIA(Attack):
    r"""
    attack based on PGD (Linf)

    Arguments for PGD:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    """

    def __init__(self, model, eps=0.3,
                 alpha=2 / 255, steps=40, gamma=64, random_start=True, guide_samples: List = None, use_layers: List = None):
        super().__init__("attack", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.gamma = gamma
        self.guide_samples = guide_samples
        self.use_layers = use_layers
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        mmd_loss = MMDLoss()

        adv_images = images.clone().detach()

        # Creating guide sample feats
        natural_images = self.guide_samples if self.guide_samples else [images.clone().detach()]
        guide_feats = []
        for x_g in natural_images:
            _ = self.model(x_g)
            guide_feats.append(self.model.features.copy())  # Deep copy!

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            adv_feat = self.model.features

            # Calculate perturbation loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Calculate statistical loss
            statistical_loss = torch.tensor(0.).to(self.model.device)
            # Default, use all hooked layers
            if self.use_layers is None:
                for guide_feat in guide_feats:
                    for k, v in guide_feat.items():
                        statistical_loss += mmd_loss(torch.flatten(adv_feat[k], start_dim=1), torch.flatten(v, start_dim=1))
            else: # Use only layers specified
                for guide_feat in guide_feats:
                    for k in self.use_layers:
                        statistical_loss += mmd_loss(torch.flatten(adv_feat[k], start_dim=1), torch.flatten(guide_feat[k], start_dim=1))
            statistical_loss /= len(guide_feats)

            # print('statistical loss', statistical_loss)
            # print('perturbation loss ', cost)

            cost -= self.gamma * statistical_loss

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        del guide_feats

        return adv_images
