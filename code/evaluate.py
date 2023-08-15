import sys
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import math

import Config


def m_NDCG(actual, predicted, topk):
    res = 0
    k = min(topk, len(actual))
    idcg = idcg_k(k)
    dcg_k = sum([int(predicted[j] in set(actual)) / math.log(j + 2, 2) for j in range(topk)])
    res += dcg_k / idcg
    return res / float(len(actual))


def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def m_MRR(targets, pred):
    m = 0
    for p in pred:
        if p in targets:
            index = pred.index(p)
            m += np.reciprocal(float(index + 1))
    return m / len(pred)


def compute_precision_recall(targets, predictions, k):
    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    precision = float(num_hit) / len(pred)
    recall = float(num_hit) / len(targets)
    hr = min(1, num_hit)
    mrr = m_MRR(targets, pred)
    ndcg = m_NDCG(targets, pred, k)
    # return precision, recall, hr, mrr, ndcg
    return hr, ndcg


def evaluate_pop_predict(model, loader, config):
    pred_all, pop_gt_all = torch.Tensor([]).to(config.device), torch.Tensor([]).to(config.device)
    item_all = torch.Tensor([]).to(config.device)
    for item, time_release, side_info, time, pop_history, pop_gt in loader:
        pop_history = torch.stack(pop_history, 0).transpose(0, 1)
        item_genre, item_director, item_actor = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])

        item_genre = torch.stack(side_info[0], 0).transpose(0, 1)
        if config.is_douban:
            item_director = side_info[1][0]
            item_actor = torch.stack(side_info[2], 0).transpose(0, 1)

        item = item.to(config.device)
        time_release = time_release.to(config.device)
        item_genre = item_genre.to(config.device)
        item_director = item_director.to(config.device)
        item_actor = item_actor.to(config.device)
        time = time.to(config.device)
        pop_history = pop_history.to(torch.float).to(config.device)
        pop_gt = pop_gt.to(device=config.device, dtype=torch.float)

        pred = model(item=item, time_release=time_release, item_genre=item_genre, item_director=item_director,
                     item_actor=item_actor, time=time, pop_history=pop_history, pop_gt=pop_gt)
        pred = pred.squeeze()
        item_all = torch.cat((item_all, item))
        pred_all = torch.cat((pred_all, pred))
        pop_gt_all = torch.cat((pop_gt_all, pop_gt))

    mae = torch.mean(torch.abs(pred_all-pop_gt_all)).item()
    mseloss = nn.MSELoss()
    mse = mseloss(pred_all, pop_gt_all).item()
    return mae, mse, item_all.cpu().numpy().tolist(), pred_all.cpu().numpy().tolist(), pop_gt_all.cpu().numpy().tolist()


def pop_func(x):
    if x > 0 :
        return math.log(x + 1)
    else:
        return x
    # return x

def pred_for_all_item(model, config:Config.Config):
    pred_all, pop_gt_all = torch.Tensor([]).to(config.device), torch.Tensor([]).to(config.device)

    dict_idx_item = {}
    idx_list = list(range(config.num_item))
    for key in config.dict_item_idx:
        dict_idx_item[config.dict_item_idx[key]] = key
    item_list = [dict_idx_item[i] for i in idx_list]


    time_release = [config.dict_item_time_release[i] for i in item_list]
    side_info = []
    for i in item_list:
        if i in config.dict_side_info:
            side_info.append(config.dict_side_info[i])
        else:
            side_info.append('[[0]]')

    # side_info = [config.dict_side_info[i] for i in item_list]

    # select test time
    time_now = config.max_time - config.test_time_range + 1

    time = [time_now for _ in item_list]
    length = config.pop_history_length
    pop_history = [config.dict_item_pop[i][time_now-length:time_now] for i in item_list]
    pop_gt = [config.dict_item_pop[i][time_now] for i in item_list]
    pop_gt = list(map(pop_func, pop_gt))
    pop_history = [list(map(pop_func, i)) for i in pop_history]

    side_info = list(map(eval, side_info))
    item_genre = [i[0] for i in side_info]
    item_director, item_actor = [0], [0]
    if config.is_douban:
        item_director = [i[1][0] for i in side_info]
        item_actor = [i[2] for i in side_info]

    item = torch.tensor(idx_list).to(config.device)
    time_release = torch.tensor(time_release).to(config.device)
    item_genre = torch.tensor(item_genre).to(config.device)
    item_director = torch.tensor(item_director).to(config.device)
    item_actor = torch.tensor(item_actor).to(config.device)
    time = torch.tensor(time).to(config.device)
    pop_history = torch.tensor(pop_history).to(config.device).to(torch.float)

    pop_gt = torch.tensor(pop_gt).to(config.device)


    pred = model(item=item, time_release=time_release, item_genre=item_genre, item_director=item_director,
                 item_actor=item_actor, time=time, pop_history=pop_history, pop_gt=pop_gt)
    pred = pred.squeeze()

    pred_all = torch.cat((pred_all, pred))
    pop_gt_all = torch.cat((pop_gt_all, pop_gt))

    mae = torch.mean(torch.abs(pred_all-pop_gt_all)).item()
    mseloss = nn.MSELoss()
    mse = mseloss(pred_all, pop_gt_all).item()
    return mae, mse, item.cpu().numpy().tolist(), pred_all.cpu().numpy().tolist(), pop_gt_all.cpu().numpy().tolist()


def calculate_hr_ndcg(model, config: Config.Config):
    douban_file = '/home/yinan/jing/data/tra_data/{}/{}_tra_test.txt'.format(config.ori_dataset, config.ori_dataset)
    test_df = pd.read_csv(douban_file,
        sep=' ', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    test_data = {}
    for i in tqdm(range(len(test_df))):
        user_now = test_df['user'][i]
        item_now = test_df['item'][i]
        if user_now not in test_data:
            test_data[user_now] = []
        test_data[user_now].append(item_now)
        
    pred_all, pop_gt_all = torch.Tensor([]).to(config.device), torch.Tensor([]).to(config.device)

    dict_idx_item = {}
    idx_list = list(range(config.num_item))
    for key in config.dict_item_idx:
        dict_idx_item[config.dict_item_idx[key]] = key
    item_list = [dict_idx_item[i] for i in idx_list]
    time_release = [config.dict_item_time_release[i] for i in item_list]
    side_info = []
    for i in item_list:
        if i in config.dict_side_info:
            side_info.append(config.dict_side_info[i])
        else:
            side_info.append('[[0]]')
    # select test time
    time_now = config.max_time - config.test_time_range + 1
    time = [time_now for _ in item_list]
    length = config.pop_history_length
    pop_history = [config.dict_item_pop[i][time_now-length:time_now] for i in item_list]

    pop_history = [config.dict_item_pop[item][config.dict_item_time_release[item]: time_now] for item in item_list]
    for i in range(len(pop_history)):
        if len(pop_history[i]) == 0:
            pop_history[i] = [0]
    valid_pop_len = [len(line) for line in pop_history]
    for i in range(len(pop_history)):
        pop_history[i] += [-1 for _ in range(config.max_time - valid_pop_len[i] + 1)]

    pop_gt = [config.dict_item_pop[i][time_now] for i in item_list]
    pop_gt = list(map(pop_func, pop_gt))
    pop_history = [list(map(pop_func, i)) for i in pop_history]
    side_info = list(map(eval, side_info))
    item_genre = [i[0] for i in side_info]
    item_director, item_actor = [0], [0]
    if config.is_douban:
        item_director = [i[1][0] for i in side_info]
        item_actor = [i[2] for i in side_info]

    item = torch.tensor(idx_list).to(config.device)
    time_release = torch.tensor(time_release).to(config.device)
    item_genre = torch.tensor(item_genre).to(config.device)
    item_director = torch.tensor(item_director).to(config.device)
    item_actor = torch.tensor(item_actor).to(config.device)
    time = torch.tensor(time).to(config.device)
    pop_history = torch.tensor(pop_history).to(config.device).to(torch.float)

    if config.alpha > 0:
        valid_pop_len = torch.clamp(valid_pop_len, max=config.alpha)
    valid_pop_len = torch.tensor(valid_pop_len).to(config.device).to(torch.float)

    pop_gt = torch.tensor(pop_gt).to(config.device)

    pop_history_output, time_output, sideinfo_output, periodic_output,  pred = model(
        item=item, time_release=time_release, item_genre=item_genre, item_director=item_director,
        item_actor=item_actor, time=time, pop_history=pop_history, pop_gt=pop_gt, valid_pop_len=valid_pop_len)
    pred = pred.squeeze()

    _, prediction = torch.topk(pred, 10)
    prediction = prediction.cpu().numpy().tolist()
    HR, NDCG = [], []
    for i in test_data.keys():
        gt_now = test_data[i]
        list_r, list_n = [], []
        for k in [5, 10]:
            r, n = compute_precision_recall(gt_now, prediction, k)
            list_r.append(r)
            list_n.append(n)
        HR.append(list_r)
        NDCG.append(list_n)

    mse = []
    mseloss = nn.MSELoss()
    mse.append(mseloss(pop_history_output.squeeze(), pop_gt).item())
    mse.append(mseloss(time_output.squeeze(), pop_gt).item())
    mse.append(mseloss(sideinfo_output.squeeze(), pop_gt).item())
    mse.append(mseloss(periodic_output.squeeze(), pop_gt).item())
    mse.append(mseloss(pred.squeeze(), pop_gt).item())
    return np.mean(HR, axis=0), np.mean(NDCG, axis=0), mse, pred.cpu().numpy().tolist()




