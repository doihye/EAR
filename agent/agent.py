import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import utils
from utils import add_random_boxes_cuda
from agent.encoder import make_encoder, TransitionModel
import cv2
import copy


LOG_FREQ = 10000


def make_agent(obs_shape, action_shape, args):
    return SacSSAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        hidden_dim=args.hidden_dim,
        discount=args.discount,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        alpha_beta=args.alpha_beta,
        actor_lr=args.actor_lr,
        actor_beta=args.actor_beta,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        actor_update_freq=args.actor_update_freq,
        critic_lr=args.critic_lr,
        critic_beta=args.critic_beta,
        critic_tau=args.critic_tau,
        critic_target_update_freq=args.critic_target_update_freq,
        encoder_feature_dim=args.encoder_feature_dim,
        encoder_lr=args.encoder_lr,
        encoder_tau=args.encoder_tau,
        use_rot=args.use_rot,
        use_inv=args.use_inv,
        use_curl=args.use_curl,
        ss_lr=args.ss_lr,
        ss_update_freq=args.ss_update_freq,
        num_layers=args.num_layers,
        num_shared_layers=args.num_shared_layers,
        num_filters=args.num_filters,
        curl_latent_dim=args.curl_latent_dim,
    )


def gaussian_logprob(noise, log_std):
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    def __init__(
        self, obs_shape, action_shape, hidden_dim,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters, num_shared_layers
    ):
        super().__init__()

        self.encoder = make_encoder(
            obs_shape, encoder_feature_dim, num_layers,
            num_filters, num_shared_layers
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
        self.apply(weight_init)

    def color_jitter_(self, img, perm, b, c, s, h):
        img1 = img[:, 0:3]
        img2 = img[:, 3:6]
        img3 = img[:, 6:9]

        for fn_id in perm:
            if fn_id == 0:
                img1 = TF.adjust_brightness(img1, b)
                img2 = TF.adjust_brightness(img2, b)
                img3 = TF.adjust_brightness(img3, b)
            elif fn_id == 1:
                img1 = TF.adjust_contrast(img1, c)
                img2 = TF.adjust_contrast(img2, c)
                img3 = TF.adjust_contrast(img3, c)
            elif fn_id == 2:
                img1 = TF.adjust_saturation(img1, s)
                img2 = TF.adjust_saturation(img2, s)
                img3 = TF.adjust_saturation(img3, s)
            elif fn_id == 3:
                img1 = TF.adjust_hue(img1, h)
                img2 = TF.adjust_hue(img2, h)
                img3 = TF.adjust_hue(img3, h)

        img = torch.cat((img1, img2, img3), 1)

        return img

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False, add_only_aug=False):

        dg = 1
        obs_aug = obs.clone().detach()

        if add_only_aug:

            random_aug1 = random.randint(0, 1)

            if random_aug1:
                zeros1 = torch.zeros_like(obs_aug[0::2, :, :, :])
                random_boxes1 = add_random_boxes_cuda(zeros1, boxes=50 * dg, mu=0, sigma=2 * dg, size_min=1,
                                                      size_max=800 * dg)
                obs_aug[0::2, :, :, :] += random_boxes1
            else:
                perm1 = torch.randperm(4)
                b_1 = random.gauss(1, 0.5 * dg)
                c_1 = random.gauss(1, 0.5 * dg)
                s_1 = random.gauss(1, 0.5 * dg)
                h_1 = random.gauss(0, 0.4 * dg)

                b_1 = np.clip(b_1, 0.5, 2)
                c_1 = np.clip(c_1, 0.5, 2)
                s_1 = np.clip(s_1, 0.5, 5)
                h_1 = np.clip(h_1, 0, 0.5)

                obs_aug[0::2] = self.color_jitter_(obs_aug[0::2] / 255, perm1, b_1, c_1, s_1, h_1) * 255

            random_aug2 = random.randint(0, 1)

            if random_aug2:
                zeros2 = torch.zeros_like(obs_aug[1::2, :, :, :])
                random_boxes2 = add_random_boxes_cuda(zeros2, boxes=50 * dg, mu=0, sigma=2 * dg, size_min=1,
                                                      size_max=800 * dg)
                obs_aug[1::2, :, :, :] += random_boxes2
            else:
                perm2 = torch.randperm(4)
                b_2 = random.gauss(1, 0.5 * dg)
                c_2 = random.gauss(1, 0.5 * dg)
                s_2 = random.gauss(1, 0.5 * dg)
                h_2 = random.gauss(0, 0.4 * dg)

                b_2 = np.clip(b_2, 0.5, 2)
                c_2 = np.clip(c_2, 0.5, 2)
                s_2 = np.clip(s_2, 0.5, 5)
                h_2 = np.clip(h_2, 0, 0.5)

                obs_aug[1::2] = self.color_jitter_(obs_aug[1::2] / 255, perm2, b_2, c_2, s_2, h_2) * 255

        obs_aug, _, _ = self.encoder(obs_aug, obs_aug, detach=detach_encoder) # -> here obs 3, 100  -> it should be 1, 100!!!!

        mu, log_std = self.trunk(obs_aug).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class RotFunction(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, h):
        return self.trunk(h)


class InvFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(2*obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, h, h_next):
        joint_h = torch.cat([h, h_next], dim=1)
        return self.trunk(joint_h)


class CURL(nn.Module):
    def __init__(self, obs_shape, z_dim, batch_size, critic, critic_target, output_type="continuous"):
        super(CURL, self).__init__()
        self.batch_size = batch_size
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder 
        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        Wz = torch.matmul(self.W, z_pos.T)
        logits = torch.matmul(z_a, Wz)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class Critic(nn.Module):
    def __init__(
        self, obs_shape, action_shape, hidden_dim,
        encoder_feature_dim, num_layers, num_filters, num_shared_layers
    ):
        super().__init__()

        self.encoder = make_encoder(
            obs_shape, encoder_feature_dim, num_layers,
            num_filters, num_shared_layers
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.apply(weight_init)

    def color_jitter_(self, img, perm, b, c, s, h):
        img1 = img[:, 0:3]
        img2 = img[:, 3:6]
        img3 = img[:, 6:9]

        for fn_id in perm:
            if fn_id == 0:
                img1 = TF.adjust_brightness(img1, b)
                img2 = TF.adjust_brightness(img2, b)
                img3 = TF.adjust_brightness(img3, b)
            elif fn_id == 1:
                img1 = TF.adjust_contrast(img1, c)
                img2 = TF.adjust_contrast(img2, c)
                img3 = TF.adjust_contrast(img3, c)
            elif fn_id == 2:
                img1 = TF.adjust_saturation(img1, s)
                img2 = TF.adjust_saturation(img2, s)
                img3 = TF.adjust_saturation(img3, s)
            elif fn_id == 3:
                img1 = TF.adjust_hue(img1, h)
                img2 = TF.adjust_hue(img2, h)
                img3 = TF.adjust_hue(img3, h)

        img = torch.cat((img1, img2, img3), 1)

        return img

    def forward(self, obs, action, detach_encoder=False, add_only_aug=False, is_target=False):

        dg = 1
        obs_aug = obs.clone().detach()

        if is_target:
            dg = 0.2

        if add_only_aug:

            random_aug1 = random.randint(0, 1)

            if random_aug1:
                zeros1 = torch.zeros_like(obs_aug[0::2, :, :, :])
                random_boxes1 = add_random_boxes_cuda(zeros1, boxes=50 * dg, mu=0, sigma=2 * dg, size_min=1, size_max=800 * dg)
                obs_aug[0::2, :, :, :] += random_boxes1
            else:
                perm1 = torch.randperm(4)
                b_1 = random.gauss(1, 0.5 * dg)
                c_1 = random.gauss(1, 0.5 * dg)
                s_1 = random.gauss(1, 0.5 * dg)
                h_1 = random.gauss(0, 0.4 * dg)

                b_1 = np.clip(b_1, 0.5, 2)
                c_1 = np.clip(c_1, 0.5, 2)
                s_1 = np.clip(s_1, 0.5, 5)
                h_1 = np.clip(h_1, 0, 0.5)

                obs_aug[0::2] = self.color_jitter_(obs_aug[0::2] / 255, perm1, b_1, c_1, s_1, h_1) * 255

            random_aug2 = random.randint(0, 1)

            if random_aug2:
                zeros2 = torch.zeros_like(obs_aug[1::2, :, :, :])
                random_boxes2 = add_random_boxes_cuda(zeros2, boxes=50 * dg, mu=0, sigma=2 * dg, size_min=1, size_max=800 * dg)
                obs_aug[1::2, :, :, :] += random_boxes2
            else:
                perm2 = torch.randperm(4)
                b_2 = random.gauss(1, 0.5 * dg)
                c_2 = random.gauss(1, 0.5 * dg)
                s_2 = random.gauss(1, 0.5 * dg)
                h_2 = random.gauss(0, 0.4 * dg)

                b_2 = np.clip(b_2, 0.5, 2)
                c_2 = np.clip(c_2, 0.5, 2)
                s_2 = np.clip(s_2, 0.5, 5)
                h_2 = np.clip(h_2, 0, 0.5)

                obs_aug[1::2] = self.color_jitter_(obs_aug[1::2] / 255, perm2, b_2, c_2, s_2, h_2) * 255

        obs_aug, _, _ = self.encoder(obs_aug, obs_aug, detach=detach_encoder)

        q1 = self.Q1(obs_aug, action)
        q2 = self.Q2(obs_aug, action)

        return q1, q2


class SacSSAgent(object):
    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        use_rot=False,
        use_inv=False,
        use_curl=False,
        ss_lr=1e-3,
        ss_update_freq=1,
        num_layers=4,
        num_shared_layers=4,
        num_filters=32,
        curl_latent_dim=128,
    ):
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.ss_update_freq = ss_update_freq
        self.use_rot = use_rot
        self.use_inv = use_inv
        self.use_curl = use_curl
        self.curl_latent_dim = curl_latent_dim

        assert num_layers >= num_shared_layers, 'num shared layers cannot exceed total amount'

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, num_layers
        ).cuda()

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim, num_layers, num_filters, num_layers
        ).cuda()

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim, num_layers, num_filters, num_layers
        ).cuda()

        self.dynamics_model = TransitionModel(channels=32,
                                              num_actions=action_shape[0],
                                              pixels=32 * 32,
                                              hidden_size=num_filters * obs_shape[0] // 3,
                                              limit=1,
                                              blocks=0,
                                              norm_type='bn',
                                              renormalize=1,
                                              residual=0.).cuda()

        self.prediction = nn.Linear(encoder_feature_dim, encoder_feature_dim).cuda()

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        self.rot = None
        self.inv = None
        self.curl = None
        self.ss_encoder = None

        if use_rot or use_inv:
            self.ss_encoder = make_encoder(
                obs_shape, encoder_feature_dim, num_layers,
                num_filters, num_shared_layers
            ).cuda()
            self.ss_encoder.copy_conv_weights_from(self.critic.encoder, num_shared_layers)

            if use_rot:
                self.rot = RotFunction(encoder_feature_dim, hidden_dim).cuda()
                self.rot.apply(weight_init)

            if use_inv:
                self.inv = InvFunction(encoder_feature_dim, action_shape[0], hidden_dim).cuda()
                self.inv.apply(weight_init)

        if use_curl:
            self.curl = CURL(obs_shape, encoder_feature_dim,
                self.curl_latent_dim, self.critic, self.critic_target, output_type='continuous').cuda()

        self.init_ss_optimizers(encoder_lr, ss_lr)

        self.spr_optimizer = torch.optim.Adam(
            list(self.dynamics_model.parameters()) + list(self.prediction.parameters()), lr=encoder_lr
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def _load_video(self, video):
        cap = cv2.VideoCapture(video)
        assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) >= 100, 'width must be at least 100 pixels'
        assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) >= 100, 'height must be at least 100 pixels'
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        buf = torch.zeros((n, 84, 84, 3))
        i, ret = 0, True
        while (i < n and ret):
            ret, frame = cap.read()
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dsize=(84, 84))
            buf[i] = torch.from_numpy(frame)
            i += 1
        cap.release()
        return buf.permute((0, 3, 1, 2)).cuda()

    def init_ss_optimizers(self, encoder_lr=1e-3, ss_lr=1e-3):
        if self.ss_encoder is not None:
            self.encoder_optimizer =  torch.optim.Adam(
                self.ss_encoder.parameters(), lr=encoder_lr
            )
        if self.use_rot:
            self.rot_optimizer =  torch.optim.Adam(
                self.rot.parameters(), lr=ss_lr
            )
        if self.use_inv:
            self.inv_optimizer =  torch.optim.Adam(
                self.inv.parameters(), lr=ss_lr
            )
        if self.use_curl:
            self.encoder_optimizer =  torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )
            self.curl_optimizer =  torch.optim.Adam(
                self.curl.parameters(), lr=ss_lr
            )
    
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.dynamics_model.train(training)
        self.prediction.train(training)
        if self.ss_encoder is not None:
            self.ss_encoder.train(training)
        if self.rot is not None:
            self.rot.train(training)
        if self.inv is not None:
            self.inv.train(training)
        if self.curl is not None:
            self.curl.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).cuda()
            obs = obs.unsqueeze(0)
            mu, _, _, _, = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).cuda()
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs, add_only_aug=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action, add_only_aug=True, is_target=True)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action, add_only_aug=True)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True, add_only_aug=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True, add_only_aug=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()# + Mutual_loss + recon_loss

        if L is not None:
            L.log('train_actor/loss', actor_loss, step)
            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                                ) + log_std.sum(dim=-1)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log('train_alpha/loss', alpha_loss, step)
                L.log('train_alpha/value', self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update_rot(self, obs, L=None, step=None):
        assert obs.shape[-1] == 84

        obs, label = utils.rotate(obs)
        h = self.ss_encoder(obs)
        pred_rotation = self.rot(h)
        rot_loss = F.cross_entropy(pred_rotation, label)

        self.encoder_optimizer.zero_grad()
        self.rot_optimizer.zero_grad()
        rot_loss.backward()

        self.encoder_optimizer.step()
        self.rot_optimizer.step()

        if L is not None:
            L.log('train_rot/rot_loss', rot_loss, step)

        return rot_loss.item()

    def rgb2hsv_torch(self, rgb: torch.Tensor) -> torch.Tensor:
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
        cmax_idx[delta == 0] = 3
        hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsv_h[cmax_idx == 3] = 0.
        hsv_h /= 6.
        hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
        hsv_v = cmax
        return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

    def do_green_screen(self, x, bg):
        x_h, x_w = x.shape[2:]

        for x in range(x_w):
            for y in range(x_h):
                r, g, b, a = pix[x, y]
                h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255., g / 255., b / 255.)
                h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)

                min_h, min_s, min_v = (100, 80, 70)
                max_h, max_s, max_v = (185, 255, 255)
                if min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v:
                    pix[x, y] = bg[x, y]

        x = np.moveaxis(np.array(im).astype(np.uint8), -1, 0)[:3]

        return x

    def color_jitter_(self, img, perm, b, c, s, h):
        img1 = img[:, 0:3]
        img2 = img[:, 3:6]
        img3 = img[:, 6:9]

        for fn_id in perm:
            if fn_id == 0:
                img1 = TF.adjust_brightness(img1, b)
                img2 = TF.adjust_brightness(img2, b)
                img3 = TF.adjust_brightness(img3, b)
            elif fn_id == 1:
                img1 = TF.adjust_contrast(img1, c)
                img2 = TF.adjust_contrast(img2, c)
                img3 = TF.adjust_contrast(img3, c)
            elif fn_id == 2:
                img1 = TF.adjust_saturation(img1, s)
                img2 = TF.adjust_saturation(img2, s)
                img3 = TF.adjust_saturation(img3, s)
            elif fn_id == 3:
                img1 = TF.adjust_hue(img1, h)
                img2 = TF.adjust_hue(img2, h)
                img3 = TF.adjust_hue(img3, h)

        img = torch.cat((img1, img2, img3), 1)

        return img

    def update_inv(self, obs, next_obs, action, add_aug=False, L=None, step=None, is_eval=False):
        assert obs.shape[-1] == 84 and next_obs.shape[-1] == 84

        spr_loss = 0

        recon_obs = copy.deepcopy(obs)
        dg = 1

        if add_aug:

            random_aug1 = random.randint(0, 1)

            if random_aug1:
                zeros1 = torch.zeros_like(obs[0::2, :, :, :])
                random_boxes1 = add_random_boxes_cuda(zeros1, boxes=50, mu=0, sigma=2, size_min=1, size_max=800)
                obs[0::2, :, :, :] += random_boxes1
                recon_obs[1::2, :, :, :] += random_boxes1
            else:
                perm1 = torch.randperm(4)
                b_1 = random.gauss(1, 0.5 * dg)
                c_1 = random.gauss(1, 0.5 * dg)
                s_1 = random.gauss(1, 0.5 * dg)
                h_1 = random.gauss(0, 0.4 * dg)
                b_1 = np.clip(b_1, 0.5, 2)
                c_1 = np.clip(c_1, 0.5, 2)
                s_1 = np.clip(s_1, 0.5, 5)
                h_1 = np.clip(h_1, 0, 0.5)
                obs[0::2] = self.color_jitter_(obs[0::2] / 255, perm1, b_1, c_1, s_1, h_1) * 255
                recon_obs[1::2] = self.color_jitter_(recon_obs[1::2] / 255, perm1, b_1, c_1, s_1, h_1) * 255

            random_aug2 = random.randint(0, 1)

            if random_aug2:
                zeros2 = torch.zeros_like(obs[1::2, :, :, :])
                random_boxes2 = add_random_boxes_cuda(zeros2, boxes=50, mu=0, sigma=2, size_min=1, size_max=800)
                obs[1::2, :, :, :] += random_boxes2
                recon_obs[0::2, :, :, :] += random_boxes2
            else:
                perm2 = torch.randperm(4)
                b_2 = random.gauss(1, 0.5 * dg)
                c_2 = random.gauss(1, 0.5 * dg)
                s_2 = random.gauss(1, 0.5 * dg)
                h_2 = random.gauss(0, 0.4 * dg)
                b_2 = np.clip(b_2, 0.5, 2)
                c_2 = np.clip(c_2, 0.5, 2)
                s_2 = np.clip(s_2, 0.5, 5)
                h_2 = np.clip(h_2, 0, 0.5)
                obs[1::2] = self.color_jitter_(obs[1::2] / 255, perm2, b_2, c_2, s_2, h_2) * 255
                recon_obs[0::2] = self.color_jitter_(recon_obs[0::2] / 255, perm2, b_2, c_2, s_2, h_2) * 255

        if add_aug:
            nobs_latent = self.critic_target.encoder.outputs['ln']

        h_next, _, _ = self.ss_encoder(next_obs, next_obs, add_aug=add_aug)

        if is_eval:
            with torch.no_grad():
                nobs_latent = self.ss_encoder.outputs['ln']

        h, Mutual_loss, recon_loss = self.ss_encoder(obs, recon_obs, add_aug=add_aug)

        if add_aug:
            obs_feature = self.ss_encoder.outputs['age']

            pred_obs = self.dynamics_model(obs_feature, action)
            pred_obs_latent = self.projection(pred_obs, self.ss_encoder)

            spr_loss = self.global_spr_loss(pred_obs_latent, nobs_latent)

        if is_eval:
            obs_feature = self.ss_encoder.outputs['age']

            pred_obs = self.dynamics_model(obs_feature, action)
            pred_obs_latent = self.projection(pred_obs, self.ss_encoder)

            spr_loss = self.global_spr_loss(pred_obs_latent, nobs_latent)

        pred_action = self.inv(h, h_next)
        inv_loss = F.mse_loss(pred_action, action) + Mutual_loss * 0.01 + recon_loss * 1

        loss = inv_loss + spr_loss * 10

        self.encoder_optimizer.zero_grad()
        self.inv_optimizer.zero_grad()
        self.spr_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.inv_optimizer.step()
        self.spr_optimizer.step()

        if L is not None:
            L.log('train_inv/inv_loss', inv_loss - Mutual_loss * 0.01 - recon_loss * 1, step)
            L.log('train_mutual/loss', Mutual_loss * 0.01, step)
            L.log('train_recon/loss', recon_loss * 1, step)
            L.log('train_spr/loss', spr_loss * 10, step)

        return loss.item()

    def update_curl(self, obs_anchor, obs_pos, L=None, step=None, ema=False):
        assert obs_anchor.shape[-1] == 84 and obs_pos.shape[-1] == 84

        z_a = self.curl.encode(obs_anchor)
        z_pos = self.curl.encode(obs_pos, ema=True)
        
        logits = self.curl.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().cuda()
        curl_loss = F.cross_entropy(logits, labels)
        
        self.encoder_optimizer.zero_grad()
        self.curl_optimizer.zero_grad()
        curl_loss.backward()

        self.encoder_optimizer.step()
        self.curl_optimizer.step()
        if L is not None:
            L.log('train/curl_loss', curl_loss, step)

        if ema:
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        return curl_loss.item()


    def update(self, replay_buffer, L, step, add_aug=False, is_eval=False):
        if self.use_curl:
            obs, action, reward, next_obs, not_done, curl_kwargs = replay_buffer.sample_curl()
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample()
        
        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )
        
        if self.rot is not None and step % self.ss_update_freq == 0:
            self.update_rot(obs, L, step)

        if self.inv is not None and step % self.ss_update_freq == 0:
            self.update_inv(obs, next_obs, action, add_aug, L, step, is_eval)

        if self.curl is not None and step % self.ss_update_freq == 0:
            obs_anchor, obs_pos = curl_kwargs["obs_anchor"], curl_kwargs["obs_pos"]
            self.update_curl(obs_anchor, obs_pos, L, step)

    def projection(self, feature, encoder):
        latent = feature.view(feature.size(0), -1)
        latent = encoder.fc(latent)
        latent = encoder.ln(latent)
        latent = torch.tanh(latent)
        return latent

    def spr_loss(self, f_x1s, f_x2s):
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        loss = F.mse_loss(f_x1, f_x2, reduction="none").mean()
        return loss

    def global_spr_loss(self, latents, target_latents):
        latents = self.prediction(latents)
        loss = self.spr_loss(latents, target_latents)
        return loss

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.dynamics_model.state_dict(), '%s/dynamics_model_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.prediction.state_dict(), '%s/prediction_model_%s.pt' % (model_dir, step)
        )
        if self.rot is not None:
            torch.save(
                self.rot.state_dict(),
                '%s/rot_%s.pt' % (model_dir, step)
            )
        if self.inv is not None:
            torch.save(
                self.inv.state_dict(),
                '%s/inv_%s.pt' % (model_dir, step)
            )
        if self.curl is not None:
            torch.save(
                self.curl.state_dict(),
                '%s/curl_%s.pt' % (model_dir, step)
            )
        if self.ss_encoder is not None:
            torch.save(
                self.ss_encoder.state_dict(),
                '%s/ss_encoder_%s.pt' % (model_dir, step)
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.dynamics_model.load_state_dict(
            torch.load('%s/dynamics_model_%s.pt' % (model_dir, step))
        )
        self.prediction.load_state_dict(
            torch.load('%s/prediction_model_%s.pt' % (model_dir, step))
        )
        if self.rot is not None:
            self.rot.load_state_dict(
                torch.load('%s/rot_%s.pt' % (model_dir, step))
            )
        if self.inv is not None:
            self.inv.load_state_dict(
                torch.load('%s/inv_%s.pt' % (model_dir, step))
            )
        if self.curl is not None:
            self.curl.load_state_dict(
                torch.load('%s/curl_%s.pt' % (model_dir, step))
            )
        if self.ss_encoder is not None:
            self.ss_encoder.load_state_dict(
                torch.load('%s/ss_encoder_%s.pt' % (model_dir, step))
            )
