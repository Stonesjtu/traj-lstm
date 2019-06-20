import torch
from torch import nn

class TrajLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        time_lstm = [nn.LSTM(input_size, hidden_size)]
        for _ in range(num_layers - 1):
            time_lstm.append(nn.LSTM(hidden_size, hidden_size))
        self.time_lstm = nn.Sequential(*time_lstm)
        self.drop = nn.Dropout(dropout)
        self.depth_lstm = nn.LSTM(hidden_size, hidden_size)
        # self.layernorm = torch.nn.LayerNorm(hidden_size)

    def forward(self, input, hidden=None):
        time_output = input
        time_results = []
        if hidden is None:
            hidden = [None for _ in self.time_lstm]
        next_hidden = []
        next_cell = []
        all_h, all_c = hidden
        for lstm, h, c in zip(self.time_lstm, all_h, all_c):

            state = (h.unsqueeze(0), c.unsqueeze(0))
            time_output, (next_h, next_c) = lstm(time_output, state)  # seq X bs X hidden
            next_hidden.append(next_h)
            next_cell.append(next_c)
            time_output = self.drop(time_output)
            # time_output = self.layernorm(time_output)
            time_results.append(time_output)

        time_results = torch.stack(time_results)  # depth X seq X bs X hidden
        depth, seq, bs, hidden = time_results.size()
        depth_results, _ = self.depth_lstm(time_results.view(depth, seq*bs, hidden))
        output = depth_results[-1]  # seq*bs X hidden
        output = output.view(seq, bs, hidden) + time_output
        next_state = (torch.stack(next_hidden[::-1]).squeeze(1), torch.stack(next_cell[::-1]).squeeze(1))
        return output, next_state

    def flatten_parameters(self):
        for lstm in self.time_lstm:
            lstm.flatten_parameters()
        self.depth_lstm.flatten_parameters()

class HuhuLSTM(TrajLSTM):
    def forward(self, input, hidden=None):
        time_output = input
        hidden = [None for _ in self.time_lstm]
        next_hidden = []
        depth_next = None
        for lstm, cur_hidden in zip(self.time_lstm, hidden):
            time_output, _ = lstm(time_output, None)  # seq X bs X hidden
            time_output = self.drop(time_output)
            depth_out, depth_next = self.depth_lstm(
                time_output.view(1, -1, self.hidden_size),
                depth_next,
            )
            # use the output of time_lstm as the input of the next layer of time lstm
            time_output = depth_out.view_as(time_output)

        return time_output, None

def test_traj_lstm():
    traj_lstm = TrajLSTM(256, 256, 6).cuda()
    fake_input = torch.zeros(30, 30, 256).cuda()
    print(traj_lstm(fake_input).mean().item())

if __name__ == '__main__':
    test_traj_lstm()
