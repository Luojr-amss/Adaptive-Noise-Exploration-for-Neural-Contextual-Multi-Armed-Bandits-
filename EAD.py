from packages import *


class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class EAD:
    def __init__(self, dim, lamdba=1, nu=1, hidden=100):
        self.func = Network(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.arm_history_reward = {}  # 用于存储每个手臂的选择次数
        self.lamdba = lamdba
        self.nu = nu
        self.lr = 0.01
        self.t = 0

    def select(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        mu = self.func(tensor)
        sampled = [fx.item() for fx in mu]
        arm = np.argmax(sampled)
        return arm

    def update(self, context, reward, arm):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.arm_history_reward[arm] = self.arm_history_reward.get(arm, 0) + 1


        arm_selection_count = self.arm_history_reward[arm]
        noise_scale = max(0.01, min(0.1 / np.log(arm_selection_count + 1), 0.1))
        noisy_reward = reward + np.random.randn() * noise_scale
        self.reward.append(noisy_reward)

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
                    return tot_loss / 2000
            if batch_loss / length <= 1e-3:
                return batch_loss / length

