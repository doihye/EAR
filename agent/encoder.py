import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIM = {2: 39, 4: 35, 6: 31, 8: 27, 10: 23, 11: 21, 12: 19}

def tie_weights(src, trg):
	assert type(src) == type(trg)
	trg.weight = src.weight
	trg.bias = src.bias

class CenterCrop(nn.Module):
	def __init__(self, size):
		super().__init__()
		assert size == 84
		self.size = size

	def forward(self, x):
		assert x.ndim == 4, 'input must be a 4D tensor'
		if x.size(2) == self.size and x.size(3) == self.size:
			return x
		elif x.size(-1) == 100:
			return x[:, :, 8:-8, 8:-8]
		else:
			return ValueError('unexepcted input size')

class NormalizeImg(nn.Module):
	def forward(self, x):
		return x/255.

class PixelEncoder(nn.Module):
	def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, num_shared_layers=4):
		super().__init__()
		assert len(obs_shape) == 3

		self.feature_dim = feature_dim
		self.num_layers = num_layers
		self.num_shared_layers = num_shared_layers

		self.preprocess = nn.Sequential(
			CenterCrop(size=84), NormalizeImg()
		)

		self.convs = nn.ModuleList(
			[nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
		)
		for i in range(num_layers - 1):
			self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

		out_dim = OUT_DIM[num_layers]
		self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
		self.ln = nn.LayerNorm(self.feature_dim)

		self.di = nn.Sequential(
			nn.Conv2d(32, 16, 3, stride=1, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=False),
			nn.Dropout(p=0.5),
			nn.Conv2d(16, 16, 3, stride=1, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=False),
			nn.Dropout(p=0.5),
			nn.Conv2d(16, 32, 3, stride=1, padding=1),
			nn.BatchNorm2d(32),
		)

		self.outputs = dict()

	def forward_conv(self, obs, recon_obs, detach=False, add_aug=False):

		Mutual_loss = 0
		recon_loss = 0
		obs = self.preprocess(obs)
		recon_obs = self.preprocess(recon_obs)

		conv = torch.relu(self.convs[0](obs))
		recon_conv = torch.relu(self.convs[0](recon_obs))

		for i in range(1, self.num_layers):
			conv = torch.relu(self.convs[i](conv))
			recon_conv = torch.relu(self.convs[i](recon_conv))

		base_feature = conv
		recon_base_feature = recon_conv

		if detach:
			base_feature = base_feature.detach()
			recon_base_feature = recon_base_feature.detach()

		agent_feature = self.di(base_feature)

		self.outputs['age'] = agent_feature

		if add_aug:
			recon_agent_feature = self.di(recon_base_feature)

			env_feature = base_feature - agent_feature
			recon_env_feature = recon_base_feature = recon_agent_feature

			env_a = env_feature[0::2]
			env_b = env_feature[1::2]

			age_t = agent_feature[0::2]
			age_t2 = agent_feature[1::2]

			re_b_t = recon_base_feature[0::2]
			re_a_t2 = recon_base_feature[1::2]

			re_env_b = recon_env_feature[0::2]
			re_env_a = recon_env_feature[1::2]

			re_age_t = recon_agent_feature[0::2]
			re_age_t2 = recon_agent_feature[1::2]

			b_t = env_b + age_t
			a_t2 = env_a + age_t2

			recon_loss_1 = F.mse_loss(b_t, re_b_t)
			recon_loss_2 = F.mse_loss(a_t2, re_a_t2)

			recon_spe_loss_1 = F.mse_loss(re_env_a, env_a)
			recon_spe_loss_2 = F.mse_loss(re_env_b, env_b)
			recon_spe_loss_3 = F.mse_loss(age_t, re_age_t)
			recon_spe_loss_4 = F.mse_loss(age_t2, re_age_t2)

			recon_loss = recon_loss_1 + recon_loss_2 + (recon_spe_loss_1 + recon_spe_loss_2 + recon_spe_loss_3 + recon_spe_loss_4) * 0.1

			wh = agent_feature.size()[2]
			Mutual_agent = F.avg_pool2d(agent_feature, (wh, wh))[:, :, 0, 0]
			Mutual_env = F.avg_pool2d(env_feature, (wh, wh))[:, :, 0, 0]
			Mutual_agent = F.normalize(Mutual_agent, dim=1)
			Mutual_env = F.normalize(Mutual_env, dim=1)
			Mutual_loss = Mutual_agent * Mutual_env
			Mutual_loss = torch.abs(torch.sum(Mutual_loss, dim=1))
			Mutual_loss = torch.mean(Mutual_loss)

		h = agent_feature.view(agent_feature.size(0), -1)

		return h, Mutual_loss, recon_loss

	def forward(self, obs, recon_obs, detach=False, add_aug=False):
		h, Mutual_loss, recon_loss = self.forward_conv(obs, recon_obs, detach, add_aug) # h (3, 14112)

		h_fc = self.fc(h)
		h_norm = self.ln(h_fc)
		out = torch.tanh(h_norm)

		self.outputs['ln'] = out

		return out, Mutual_loss, recon_loss

	def copy_conv_weights_from(self, source, n=None):
		if n is None:
			n = self.num_layers
		for i in range(n):
			tie_weights(src=source.convs[i], trg=self.convs[i])


def make_encoder(
	obs_shape, feature_dim, num_layers, num_filters, num_shared_layers
):
	assert num_layers in OUT_DIM.keys(), 'invalid number of layers'
	if num_shared_layers == -1 or num_shared_layers == None:
		num_shared_layers = num_layers
	assert num_shared_layers <= num_layers and num_shared_layers > 0, \
		f'invalid number of shared layers, received {num_shared_layers} layers'
	return PixelEncoder(
		obs_shape, feature_dim, num_layers, num_filters, num_shared_layers
	)

def init_normalization(channels, type="bn", affine=True, one_d=False):
    assert type in ["bn", "ln", "in", "none", None]
    if type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=affine)
        else:
            return nn.BatchNorm2d(channels, affine=affine)
    elif type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=affine)
        else:
            return nn.GroupNorm(1, channels, affine=affine)
    elif type == "in":
        return nn.GroupNorm(channels, channels, affine=affine)
    elif type == "none" or type is None:
        return nn.Identity()


class Conv2dSame(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias,
                            stride=stride, padding=ka)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_type="bn"):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            init_normalization(out_channels, norm_type),
            Conv2dSame(out_channels, out_channels, 3),
            init_normalization(out_channels, norm_type),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out


class TransitionModel(nn.Module):
    def __init__(self,
                 channels,
                 num_actions,
                 args=None,
                 blocks=16,
                 hidden_size=256,
                 pixels=36,
                 limit=300,
                 action_dim=6,
                 norm_type="bn",
                 renormalize=True,
                 residual=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.args = args
        self.renormalize = renormalize
        self.residual = residual
        layers = [Conv2dSame(channels + num_actions, hidden_size, 3),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type)]
        for _ in range(blocks):
            layers.append(ResidualBlock(hidden_size,
                                        hidden_size,
                                        norm_type))
        layers.extend([Conv2dSame(hidden_size, channels, 3)])

        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x, action):
        batch_range = torch.arange(action.shape[0], device=action.device)
        action_onehot = torch.zeros(action.shape[0], action.shape[1], x.shape[-2], x.shape[-1], device=action.device)
        action_onehot[batch_range, :, :, :] = action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-2], x.shape[-1])

        stacked_image = torch.cat([x, action_onehot], 1)
        next_state = self.network(stacked_image)
        if self.residual:
            next_state = next_state + x
        next_state = F.relu(next_state)

        return next_state


class RewardPredictor(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_size=1,
                 pixels=36,
                 limit=300,
                 norm_type="bn"):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type),
                  nn.Flatten(-3, -1),
                  nn.Linear(pixels * hidden_size, 256),
                  nn.ReLU(),
                  nn.Linear(256, limit * 2 + 1)]
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):
        return self.network(x)


def renormalize(tensor, first_dim=1):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min) / (max - min)

    return flat_tensor.view(*tensor.shape)

