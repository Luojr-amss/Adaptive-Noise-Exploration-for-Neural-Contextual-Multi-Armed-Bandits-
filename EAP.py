
from packages import *


class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class EAP:
    def __init__(self, dim, lamdba=1, nu=1, hidden=100):
        self.func = Network(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.arm_history_reward = {}
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,)).to(device)
        self.nu = nu
        self.lr = 0.01
        self.t = 0

    def add_noise_to_parameters(self, arm, base_scale=0.005, min_scale=0.0005, max_scale=0.05):
        arm_selection_count = self.arm_history_reward.get(arm, 0)
        if arm_selection_count > 0:
            scale = max(min_scale, min(base_scale / np.log(arm_selection_count + 1), max_scale))
        else:
            scale = max_scale

        with torch.no_grad():
            for param in self.func.parameters():
                param.add_(torch.randn(param.size()).to(device) * scale)

    def select(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        mu = self.func(tensor)
        sampled = [fx.item() for fx in mu]
        arm = np.argmax(sampled)
        return arm

    def update(self, context, reward, arm):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)

        if arm in self.arm_history_reward:
            self.arm_history_reward[arm] += 1
        else:
            self.arm_history_reward[arm] = 1

    def train(self, t):
        optimizer = optim.SGD(self.func.parameters(), lr=self.lr)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for i, idx in enumerate(index):
                c = self.context_list[idx]
                r = self.reward[idx]

                optimizer.zero_grad()
                delta = self.func(c.to(device)) - r
                loss = delta * delta

                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 2000:

                    for arm in self.arm_history_reward.keys():
                        self.add_noise_to_parameters(arm)
                    return tot_loss / 2000
            if batch_loss / length <= 1e-3:
                for arm in self.arm_history_reward.keys():
                    self.add_noise_to_parameters(arm)
                return batch_loss / length
