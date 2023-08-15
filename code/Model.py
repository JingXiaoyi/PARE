import random

import torch
import torch.nn as nn
import numpy as np
import Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

torch.manual_seed(2023)
torch.cuda.manual_seed(2023)


class ModulePopHistory(nn.Module):
    def __init__(self, config: Config.Config):
        super(ModulePopHistory, self).__init__()
        self.config = config
        self.LSTM = nn.LSTM(input_size=1,
                            hidden_size=config.embed_size,
                            num_layers=2,
                            batch_first=True)
        self.fc_output = nn.Linear(config.pop_history_length*config.embed_size, config.embed_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.fc_output_pop_history = nn.Linear(config.embed_size, 1)

    def ema(self, pop_history, valid_pop_len):
        beta = self.config.beta
        ema_all, ema_temp = [], []
        for i in range(len(pop_history)):
            line = pop_history[i]
            for j in range(int(valid_pop_len[i].item())):
                if j == 0:
                    ema_temp.append(line[j])
                else:
                    ema_temp.append(beta * line[j] + (1 - beta) * ema_temp[j - 1])
            ema_all.append(ema_temp[-1])
        ema_all = torch.tensor(ema_all).to(self.config.device)
        return ema_all

    def forward(self, pop_history, valid_pop_len):
        history_average = self.ema(pop_history, valid_pop_len)
        # packed_input = pack_padded_sequence(pop_history.unsqueeze(-1), valid_pop_len.cpu(), batch_first=True, enforce_sorted=False)

        # pop_history_embed, _ = self.LSTM(packed_input)
        # pop_history_embed = pop_history_embed.cpu()
        # pop_history_embed, input_sizes = pad_packed_sequence(pop_history_embed, batch_first=True)
        #
        # pop_history_embed_selected = [pop_history_embed[i, input_sizes[i]-1, :].unsqueeze(0) for i in range(len(pop_history_embed))]
        # pop_history_embed_output = torch.concat(pop_history_embed_selected, dim=0).to(self.config.device)
        #
        # pop_history_embed_output = self.fc_output_pop_history(pop_history_embed_output).squeeze()
        # pop_pred = history_average + pop_history_embed_output
        # output = pop_pred.unsqueeze(-1)
        # return output
        return history_average.unsqueeze(-1)

class ModuleTime(nn.Module):
    def __init__(self, config: Config.Config):
        super(ModuleTime, self).__init__()
        self.config = config
        self.fc_item_pop_value = nn.Linear(config.embed_size*4, 1)
        self.relu = nn.ReLU()

    def forward(self, item_embed, time_release_embed, time_embed):
        temporal_dis = time_release_embed - time_embed
        item_temp_joint_embed = torch.cat((temporal_dis, item_embed, time_embed, time_release_embed), 1)
        joint_item_temp_value = self.relu(self.fc_item_pop_value(item_temp_joint_embed))
        return joint_item_temp_value


class ModuleSideInfo(nn.Module):
    def __init__(self, config: Config.Config):
        super(ModuleSideInfo, self).__init__()
        self.config = config
        if config.is_douban:
            self.fc_output = nn.Linear(3*config.embed_size, 1)
        else:
            self.fc_output = nn.Linear(config.embed_size, 1)
        self.relu = nn.ReLU()

    def forward(self, genre_embed, director_embed, actor_embed):
        genre_embed = genre_embed.mean(dim=1)
        if self.config.is_douban:
            actor_embed = actor_embed.mean(dim=1)
            embed_sideinfo = torch.cat((genre_embed, director_embed, actor_embed), 1)
            output = self.relu(self.fc_output(embed_sideinfo))
        else:
            output = self.relu(self.fc_output(genre_embed))
        return output


class ModulePeriodic(nn.Module):
    def __init__(self, config: Config.Config):
        super(ModulePeriodic, self).__init__()
        self.config = config
        # self.embed_genre_period = nn.Embedding(config.num_genre_period, config.embed_size)
        self.embed_joint_genre_time = nn.Embedding(config.num_genre_period*config.num_side_info[0], config.embed_size, padding_idx=0)
        self.fc_output = nn.Linear(config.embed_size, 1)
        self.relu = nn.ReLU()

    def forward(self, time, item_genre):
        period_time = time % self.config.num_genre_period
        period_time = period_time.unsqueeze(-1).repeat(1, item_genre.shape[-1])
        period_time_pad = torch.clamp(item_genre, max=1)
        joint_id = (item_genre * self.config.num_genre_period + period_time) * period_time_pad
        joint_embed = self.embed_joint_genre_time(joint_id).mean(dim=1)
        output = self.relu(self.fc_output(joint_embed))
        return output


class PopPredict(nn.Module):
    def __init__(self, is_training, config: Config.Config):
        super(PopPredict, self).__init__()

        self.config = config
        self.is_training = is_training
        num_genre = config.num_side_info[0]
        num_director = config.num_side_info[1] if config.is_douban else 1
        num_actor = config.num_side_info[2] if config.is_douban else 1

        self.embed_size = config.embed_size

        # embedding layer
        self.embed_item = nn.Embedding(config.num_item, self.embed_size)
        self.embed_time = nn.Embedding(config.max_time + 1, self.embed_size)
        self.embed_genre = nn.Embedding(num_genre, self.embed_size, padding_idx=0)
        self.embed_director = nn.Embedding(num_director, self.embed_size, padding_idx=0)
        self.embed_actor = nn.Embedding(num_actor, self.embed_size, padding_idx=0)

        # activate func
        self.relu = nn.ReLU()

        # modules
        self.module_pop_history = ModulePopHistory(config=config)
        self.module_periodic = ModulePeriodic(config=config)
        self.module_sideinfo = ModuleSideInfo(config=config)
        self.module_time = ModuleTime(config=config)
        self.softmax = nn.Softmax()

        self.attention_layer = nn.Linear(4, 1, bias=False)
        self.attention_layer.weight.data = torch.tensor([config.a1, config.a2, config.a3, config.a4])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if name.startswith("weight"):
                        nn.init.normal_(param, std=0.1)
                    else:
                        nn.init.zeros_(param)

    def forward(self, item, time_release, item_genre, item_director, item_actor, time, pop_history, pop_gt, valid_pop_len):
        item_embed = self.embed_item(item)
        time_release_embed = self.embed_time(time_release)
        genre_embed = self.embed_genre(item_genre)
        time_embed = self.embed_time(time)
        director_embed = torch.tensor([0]).to(self.config.device)
        actor_embed = torch.tensor([0]).to(self.config.device)
        if self.config.is_douban:
            director_embed = self.embed_director(item_director)
            actor_embed = self.embed_actor(item_actor)

        ############################### pop_history ###############################
        pop_history_output = self.module_pop_history(pop_history=pop_history, valid_pop_len=valid_pop_len)

        ################################### time ###################################
        time_output = self.module_time(item_embed=item_embed, time_release_embed=time_release_embed,
                                        time_embed=time_embed)

        ################################### periodic ###################################
        periodic_output = self.module_periodic(time=time, item_genre=item_genre)

        ################################### side ###################################
        if self.config.is_douban:
            sideinfo_output = self.module_sideinfo(genre_embed=genre_embed, director_embed=director_embed,
                                                    actor_embed=actor_embed)
        else:
            sideinfo_output = self.module_sideinfo(genre_embed=genre_embed, director_embed=None, actor_embed=None)

        pred_all = torch.cat((pop_history_output, time_output, sideinfo_output, periodic_output), 1)
        self.attention_layer.weight.data[0] = 0
        self.attention_layer.weight.data[2] = 0
        self.attention_layer.weight.data[3] = 0
        output = self.attention_layer(pred_all)/torch.sum(self.attention_layer.weight.data)
        # output = (pop_history_output + time_output + sideinfo_output + periodic_output)/4
        if not self.is_training:
            print('self.attention_layer.weight.data: {}'.format(self.attention_layer.weight.data/torch.sum(self.attention_layer.weight.data)))

        return pop_history_output, time_output, sideinfo_output, periodic_output, output
