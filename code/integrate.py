import os
import torch
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
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
    return precision, recall, hr, mrr, ndcg


def read_set(path):
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            d = line.strip().split(' ')
            data.append([int(d[0]), int(d[1]), int(float(d[2])), float(d[3])])
    return data


def make_user_seq(data):
    dict = {}
    user_seq = []
    for i in range(len(data)):
        user = data[i][0]
        if user not in dict:
            dict[user] = []
        dict[user].append(i)
    for key in dict:
        dict[key].sort(key=lambda x: data[x][3])
    for i in range(len(dict)):
        line = [data[i][1] for i in dict[i]]
        line += [0, 0]
        user_seq.append(line)

    return user_seq


def make_test_dict(data):
    dict = {}
    user_seq = []
    for i in range(len(data)):
        user = data[i][0]
        if user not in dict:
            dict[user] = []
        dict[user].append(data[i][1])
    return dict


def evaluate(gt_dict, model_result):
    PRECISION, RECALL, HR, MRR, NDCG, all_predictions = [], [], [], [], [], []
    for key in gt_dict:

        targets = gt_dict[key]
        results = model_result[key]

        predictions = np.array(results)

        predictions = np.argsort(-predictions).tolist()

        all_predictions.append(predictions[:20])
        # print(targets, predictions)
        # input('debug')

        list_p, list_r, list_h, list_m, list_n = [], [], [], [], []
        # print(key)
        # print(predictions[:20])
        # input('debug')

        for k in [1, 5, 10, 20]:
        # for k in [1, 3, 5, 7, 9, 20]:
            # precision, recall, hr, mrr, ndcg
            p, r, h, m, n = compute_precision_recall(targets, predictions, k)
            list_p.append(p)
            list_r.append(r)
            list_h.append(h)
            list_m.append(m)
            list_n.append(n)

        PRECISION.append(list_p)
        RECALL.append(list_r)
        HR.append(list_h)
        MRR.append(list_m)
        NDCG.append(list_n)
    return np.mean(PRECISION, axis=0), np.mean(RECALL, axis=0), np.mean(HR, axis=0),\
           np.mean(MRR, axis=0), np.mean(NDCG, axis=0), all_predictions


def norm(data):
    data = np.array(data)
    range = np.max(data) - np.min(data)
    data = (data - np.min(data)) / range
    return data


def calculate_mae_mse(pop_file_1, pop_file_2):
    print('read pred pop')
    pred_pop_1, pred_pop_2 = [], []
    with open('{}{}'.format(pop_result_path, pop_file_1), 'r') as f_pop:
        for line in f_pop.readlines():
            line = line.strip().split(' ')
            # pred_pop[int(float(line[0]))] = float
            pred_pop_1.append(float(line[1]))
    with open('{}{}'.format(pop_result_path, pop_file_2), 'r') as f_pop:
        for line in f_pop.readlines():
            line = line.strip().split(' ')
            # pred_pop[int(float(line[0]))] = float
            pred_pop_2.append(float(line[1]))
    mse_metric = torch.nn.MSELoss()
    pred_pop_1 = [math.log(i + 1) for i in pred_pop_1]
    pred_pop_2 = [math.log(i + 1) for i in pred_pop_2]
    tensor_1 = torch.tensor(pred_pop_1)
    tensor_2 = torch.tensor(pred_pop_2)
    mse = torch.mean(mse_metric(tensor_1, tensor_2)).item()
    mae = torch.mean(torch.abs(tensor_1 - tensor_2)).item()
    print('pop_file_1: {}'.format(pop_file_1))
    print('pop_file_2: {}'.format(pop_file_2))
    print('mae: {}'.format(mae))
    print('mse: {}'.format(mse))


def precess_result_pop(dataset, model_type, large_value_first, pop_file, model_result_file, gt_file, beta, model_num = None):
    ############################################## read pred pop #############################################
    print('read pred pop')
    pred_pop = []
    with open('{}{}'.format(pop_result_path, pop_file), 'r') as f_pop:
        for line in f_pop.readlines():
            line = line.strip().split(' ')
            # pred_pop[int(float(line[0]))] = float
            pred_pop.append(float(line[1]))

    # for i in range(len(pred_pop)):
    #     print(i, pred_pop[i])
    # exit()

    ############################################## read model output #############################################
    print('read model output')
    gt_data = read_set(gt_file)
    gt_dict = make_test_dict(gt_data)
    print('len gt_dict: {}'.format(len(gt_dict)))

    model_result = []
    if model_type == 'STOSA':
        print(model_type)
        f_model_result = '{}{}/{}'.format(model_result_path, model_type, model_result_file)
        # with open(f_model_result, 'r') as f_result:
        #     print(len(f_result.readlines()))
        #     for line in f_result.readlines():
        #         model_result.append(eval(line.strip()))
        model_result = np.load(f_model_result)
        dict_model_result = {}
        for key in gt_dict:
            dict_model_result[key] = model_result[key]
        model_result = dict_model_result
        print('filter step 1')
        for key in model_result.keys():
            model_result[key] = model_result[key][:-1]
            for j in range(len(model_result[key])):
                if model_result[key][j] > 100000:
                    model_result[key][j] = -100
        print('filter step 2')
        for key in model_result.keys():
            for j in range(len(model_result[key])):
                if model_result[key][j] == -100:
                    model_result[key][j] = max(model_result[key])

        print('len_model_result: {}'.format(len(model_result)))

    elif model_type == 'traditional_methods':
        '''
        [Random-1 2,
        TopPop-3 4,
        UserKNNCFRecommender-5,
        ItemKNNCFRecommender-6,
        SLIM_BPR_Cython-7,
        MatrixFactorization_BPR_Cython-13]
        '''
        split_file = model_result_file.split('_')
        #        []
        model_type_tra = ['none', 'Random-1', 'Random-2', 'TopPop-3', 'TopPop-4', 'UserKNN-5', 'ItemKNN-6',
                          'SLIM_BPR-7', 'MF_BPR-8', 'MF_BPR-9', 'MF_BPR-10', 'MF_BPR-11', 'MF_BPR-12', 'MF_BPR-13',
                          'MF_BPR-14', 'MF_BPR-15', 'MF_BPR-16', 'MF_BPR-17']
        model_num = str(model_num)
        print(model_type_tra[int(model_num)])
        print('model_num: {}'.format(model_num))

        user_list_file = '{}{}/{}'.format(model_result_path, model_type, '_'.join(split_file[:-3]+[model_num, 'user_list.npy']))
        model_result_file = '{}{}/{}'.format(model_result_path, model_type, '_'.join(split_file[:-3]+[model_num, 'model_result.npy']))
        user_list = np.load(user_list_file)
        model_result = np.load(model_result_file)

        dict_model_result = {}
        for i in range(len(user_list)):
            for j in range(len(model_result[i])):
                if model_result[i][j] < 0:
                    model_result[i][j] = 0
            dict_model_result[user_list[i]] = model_result[i]


        model_result = dict_model_result
        del dict_model_result

    elif 'ICLRec' in model_result_file:
        print(model_type)
        f_model_result = '{}{}/{}'.format(model_result_path, model_type, model_result_file)
        array_model_result = np.load(f_model_result)
        model_result = {}
        for i in range(len(array_model_result)):
            model_result[i] = array_model_result[i][:-1]
        print('len_model_result: {}'.format(len(model_result)))

    elif 'Caser' in model_result_file:
        print(model_type)
        f_model_result = '{}{}/{}'.format(model_result_path, model_type, model_result_file)
        model_result = {}
        with open(f_model_result, 'r') as f_read:
            for line in f_read.readlines():
                line.strip()
                user_id = int(line.split(' ', 1)[0])
                pre = line.split(' ', 1)[1]
                pre = pre.strip().split(' ')
                pre = list(map(float, pre))
                model_result[user_id] = pre
        print('len_model_result: {}'.format(len(model_result)))

    elif 'HGN' in model_result_file:
        print(model_type)
        f_model_result = '{}{}/{}'.format(model_result_path, model_type, model_result_file)
        array_model_result = np.load(f_model_result)
        model_result = {}
        for i in range(len(array_model_result)):
            model_result[i] = array_model_result[i]
        print('len_model_result: {}'.format(len(model_result)))

    elif 'SASRec' in model_result_file:
        print(model_type)
        f_model_result = '{}{}/{}'.format(model_result_path, model_type, model_result_file)
        model_result = {}
        len_item = len(pred_pop)
        print('len_pred_pop: {}'.format(len_item))
        with open(f_model_result, 'r') as f_read:
            for line in f_read.readlines():
                line = eval(line)
                user_id = int(line[0])
                pre = line[1:]
                for i in range(len(pre)):
                    if pre[i] == 'none':
                        pre[i] = -1000000
                pre = list(map(float, pre))
                pre += [-1000000 for _ in range(len_item-len(pre))]
                model_result[user_id] = pre
        for key in model_result:
            pre = model_result[key]
            max_pre = max(pre)
            for i in range(len(pre)):
                if pre[i] == -1000000:
                    pre[i] = max_pre

        print('len_model_result: {}'.format(len(model_result)))

    elif 'NCF' in model_result_file:
        print(model_type)
        f_model_result = '{}{}/{}'.format(model_result_path, model_type, model_result_file)
        with open(f_model_result, 'r') as f:
            model_result = eval(f.readlines()[0].strip())
        print(len(model_result))

    elif 'SHT' in model_result_file:
        print(model_type)
        f_user = '{}{}/{}'.format(model_result_path, model_type, 'SHT_user' + model_result_file[8:])
        f_pred = '{}{}/{}'.format(model_result_path, model_type, 'SHT_pred' + model_result_file[8:])
        a_user = np.load(f_user)
        a_pred = np.load(f_pred)
        model_result = {}
        for i in range(len(a_user)):
            model_result[a_user[i]] = a_pred[i]
        print('filtering step 1')
        for i in tqdm(model_result.keys()):
            for j in range(len(model_result[i])):
                if model_result[i][j]<-10000:
                    model_result[i][j] = 100000000
        print('filtering step 2')
        for i in tqdm(model_result.keys()):
            min_now = min(model_result[i])
            for j in range(len(model_result[i])):
                if model_result[i][j] == 100000000:
                    model_result[i][j] = min_now

    ############################################## normalization #############################################
    # do normalization and transfer model_result to dict{user_id:np.array}
    print('normalization')

    if isinstance(model_result, list):
        print('model_result is list')
        model_result = list(map(norm, model_result))
        if not large_value_first:
            model_result = [1-i for i in model_result]
        dict_model_result = {}
        for i in range(len(model_result)):
            dict_model_result[i] =model_result[i]
        model_result = dict_model_result
        del dict_model_result
    elif isinstance(model_result, dict):
        for key in model_result.keys():
            model_result[key] = norm(model_result[key])
            if not large_value_first:
                model_result[key] = 1-model_result[key]
    pred_pop = norm(pred_pop)
    print('pred_pop after norm: {}'.format(pred_pop))
    ########################################## combine pop and result #############################################
    # print(len(pred_pop))
    # exit()
    model_result_1, model_result_2 = [], []

    print('model_result is dict')
    model_result_proposed_1, model_result_proposed_2 = {}, {}

    calculate_overlap = False
    if calculate_overlap:
        ########################## calculate overlab between model_result and onlypop ##########################
        for key in model_result:
            beta = 0
            # model_result_proposed_1[key] = model_result[key] * (beta + (1 - beta) * (pred_pop))
            model_result_proposed_1[key] = model_result[key] * beta + (1-beta) * (pred_pop)
        for key in model_result:
            beta = 1
            # model_result_proposed_1[key] = model_result[key] * (beta + (1 - beta) * (pred_pop))
            model_result_proposed_2[key] = model_result[key] * beta + (1-beta) * (pred_pop)

        _, _, _, _, _, all_predictions_1 = evaluate(gt_dict, model_result_proposed_1)
        _, _, _, _, _, all_predictions_2 = evaluate(gt_dict, model_result_proposed_2)

        intersects = [[] for i in range(20)]
        for cut_off in list(range(21))[1:]:
            for i in range(len(all_predictions_1)):
                intersects[cut_off-1].append(len(np.intersect1d(all_predictions_1[i][:cut_off], all_predictions_2[i][:cut_off])))
        all_mean = [np.mean(i) for i in intersects]
        mean_overlap_file = './overlap_list/{}_{}_mean_overlap.list'.format(dataset, model_type)
        with open(mean_overlap_file, 'w') as f:
            for i in all_mean:
                f.write('{}\n'.format(i))
        # labels = list(range(21))[1:]
        # plt.figure(figsize=(12, 4))
        # plt.boxplot(intersects, labels=labels, widths=1, patch_artist=True,
        #             boxprops={'color': (0/255., 0/255., 0/255.), 'facecolor': (137/255., 128/255., 68/255.)})
        # plt.savefig('./fig/{}.jpg'.format(dataset + '_' + model_type + '_' + 'box_overlap'))
        #
        # plt.xlabel('Value')
        # plt.xlabel('Cutoff@k')
        # plt.close()

        # all_mean = [np.mean(i) for i in intersects]
        # all_std = [np.std(i) for i in intersects]
        # print('all_mean: {}'.format(all_mean))
        # print('all_std: {}'.format(all_std))
        # # print('mean: {}, std: {}'.format(np.mean(intersects), np.std(intersects)))
        # fig_x = list(range(21))[1:]
        # 
        # fig = plt.figure(figsize=(6, 4))
        # ax1 = fig.add_subplot(111)
        # line_mean = ax1.plot(fig_x, all_mean, '-', label='mean', color='cornflowerblue', marker='x', markevery=2)
        # ax2 = ax1.twinx()
        # line_std = ax2.plot(fig_x, all_std, '-', label='std', color='orange', marker='*', markevery=2)
        # 
        # lines = line_mean + line_std
        # labs = [l.get_label() for l in lines]
        # ax1.legend(lines, labs)
        # # ax1.grid()
        # # ax1.set_title('Overlap: dataset-{} model-{}'.format(dataset, model_type))
        # ax1.set_xlabel('Cutoff@k')
        # ax1.set_ylabel('Averaged overlaps')
        # # ax2.set_ylabel('STD of overlaps')
        # ax1.set_xlim(xmin=1)
        # ax1.set_ylim(ymin=0)
        # ax2.set_xlim(xmin=1)
        # ax2.set_ylim(ymin=0)
        # 
        # ax1.set_xticks(list(range(1, 20, 2)))
        # ax2.set_xticks(list(range(1, 20, 2)))
        # # ax1.xaxis.set_major_locator(plt.MultipleLocator(2))
        # # ax2.set_ylim(bottom=0.)
        # # ax2.set_ylim(bottom=0.)
        # plt.savefig('./fig/{}.pdf'.format(dataset + '_' + model_type + '_' + 'overlap'))
        # plt.savefig('./fig/{}.jpg'.format(dataset + '_' + model_type + '_' + 'overlap'))
        # plt.close()


    else:

        ##################################### normal result #####################################

        result_output = []
        best_metrics = [0, 0, 0, 0, 0, 0]
        best_lists = [[-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]]
        best_beta_list = [-1, -1, -1, -1, -1, -1]
        for beta in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # for beta in [0, 1]:
            for key in tqdm(model_result):
                # model_result_proposed_1[key] = model_result[key] * (beta + (1-beta) * (pred_pop))
                model_result_proposed_2[key] = model_result[key] * beta + (1-beta) * (pred_pop)


            PRECISION, RECALL, HR, MRR, NDCG, all_predictions_1 = \
                evaluate(gt_dict, model_result_proposed_2)
            print('\nmodel_result[key] * (beta + (1-beta) * (pred_pop))')
            '''
            print("beta: {}, HR: {:.5g} {:.5g} {:.5g} {:.5g} {:.5g} {:.5g}"
                  .format(beta, HR[0], HR[1], HR[2], HR[3], HR[4], HR[5]))
            if HR[0] > best_metrics[0]:
                best_metrics[0] = HR[0]
                best_lists[0] = [HR[0], HR[1], HR[2], HR[3], HR[4], HR[5]]
                best_beta_list[0] = beta
            if HR[1] > best_metrics[1]:
                best_metrics[1] = HR[1]
                best_lists[1] = [HR[0], HR[1], HR[2], HR[3], HR[4], HR[5]]
                best_beta_list[1] = beta
            if HR[2] > best_metrics[2]:
                best_metrics[2] = HR[2]
                best_lists[2] = [HR[0], HR[1], HR[2], HR[3], HR[4], HR[5]]
                best_beta_list[2] = beta
            if HR[3] > best_metrics[3]:
                best_metrics[3] = HR[3]
                best_lists[3] = [HR[0], HR[1], HR[2], HR[3], HR[4], HR[5]]
                best_beta_list[3] = beta
            if HR[4] > best_metrics[4]:
                best_metrics[4] = HR[4]
                best_lists[4] = [HR[0], HR[1], HR[2], HR[3], HR[4], HR[5]]
                best_beta_list[4] = beta
            if HR[5] > best_metrics[5]:
                best_metrics[5] = HR[5]
                best_lists[5] = [HR[0], HR[1], HR[2], HR[3], HR[4], HR[5]]
                best_beta_list[5] = beta
        print('\nbest hr@1: {:.5g} {:.5g} {:.5g} {:.5g} {:.5g} {:.5g}, beta: {}'.
              format(best_lists[0][0], best_lists[0][1], best_lists[0][2], best_lists[0][3], best_lists[0][4], best_lists[0][5],
                     best_beta_list[0]))
        print('\nbest hr@3: {:.5g} {:.5g} {:.5g} {:.5g} {:.5g} {:.5g}, beta: {}'.
              format(best_lists[1][0], best_lists[1][1], best_lists[1][2], best_lists[1][3], best_lists[1][4], best_lists[0][5],
                     best_beta_list[1]))
        print('\nbest hr@5: {:.5g} {:.5g} {:.5g} {:.5g} {:.5g} {:.5g}, beta: {}'.
              format(best_lists[2][0], best_lists[2][1], best_lists[2][2], best_lists[2][3], best_lists[2][4], best_lists[0][5],
                     best_beta_list[2]))
        print('\nbest hr@7: {:.5g} {:.5g} {:.5g} {:.5g} {:.5g} {:.5g}, beta: {}'.
              format(best_lists[3][0], best_lists[3][1], best_lists[3][2], best_lists[3][3], best_lists[3][4], best_lists[0][5],
                     best_beta_list[3]))
        print('\nbest hr@9: {:.5g} {:.5g} {:.5g} {:.5g} {:.5g} {:.5g}, beta: {}'.
              format(best_lists[4][0], best_lists[4][1], best_lists[4][2], best_lists[4][3], best_lists[4][4], best_lists[0][5],
                     best_beta_list[4]))
        print('\nbest hr@20: {:.5g} {:.5g} {:.5g} {:.5g} {:.5g} {:.5g}, beta: {}'.
              format(best_lists[5][0], best_lists[5][1], best_lists[5][2], best_lists[5][3], best_lists[5][4], best_lists[0][5],
                     best_beta_list[4]))
            '''
        # '''
            print("beta: {}, PRECISION: {:.5g} {:.5g} {:.5g} {:.5g} \n "
                  "RECALL: {:.5g} {:.5g} {:.5g} {:.5g} \n "
                  "HR: {:.5g} {:.5g} {:.5g} {:.5g} \n "
                  "MRR: {:.5g} {:.5g} {:.5g} {:.5g} \n "
                  "NDCG: {:.5g} {:.5g} {:.5g} {:.5g}"
                  .format(beta, PRECISION[0], PRECISION[1], PRECISION[2], PRECISION[3],
                          RECALL[0], RECALL[1], RECALL[2], RECALL[3],
                          HR[0], HR[1], HR[2], HR[3],
                          MRR[0], MRR[1], MRR[2], MRR[3],
                          NDCG[0], NDCG[1], NDCG[2], NDCG[3]))

            PRECISION, RECALL, HR, MRR, NDCG, all_predictions_2 = \
                evaluate(gt_dict, model_result_proposed_2)
            # print("beta: {}\nPRECISION: {:.5g} RECALL: {:.5g} HR: {:.5g} MRR: {:.5g} NDCG {:.5g}\n "
            #       .format(beta, PRECISION[2], RECALL[2], HR[2], MRR[2], NDCG[2]))

            result_output.append(HR.tolist())
            result_output.append(NDCG.tolist())
            if PRECISION[2]>best_metrics[0]:
                best_metrics[0] = PRECISION[2]
                best_lists[0] = [PRECISION[2], RECALL[2], HR[2], MRR[2], NDCG[2]]
                best_beta_list[0] = beta
            if RECALL[2]>best_metrics[1]:
                best_metrics[1] = RECALL[2]
                best_lists[1] = [PRECISION[2], RECALL[2], HR[2], MRR[2], NDCG[2]]
                best_beta_list[1] = beta
            if HR[2]>best_metrics[2]:
                best_metrics[2] = HR[2]
                best_lists[2] = [PRECISION[2], RECALL[2], HR[2], MRR[2], NDCG[2]]
                best_beta_list[2] = beta
            if MRR[2]>best_metrics[3]:
                best_metrics[3] = MRR[2]
                best_lists[3] = [PRECISION[2], RECALL[2], HR[2], MRR[2], NDCG[2]]
                best_beta_list[3] = beta
            if NDCG[2]>best_metrics[4]:
                best_metrics[4] = NDCG[2]
                best_lists[4] = [PRECISION[2], RECALL[2], HR[2], MRR[2], NDCG[2]]
                best_beta_list[4] = beta
        save_result = False
        if save_result:
            file_output = './pop+result_saved/{}_{}_.out'.format(dataset, model_type)
            with open(file_output, 'w') as f_out:
                f_out.write(str(result_output))
        
        print('\nbest precision: {:.5g} {:.5g} {:.5g} {:.5g} {:.5g}, beta: {}'.
              format(best_lists[0][0], best_lists[0][1], best_lists[0][2], best_lists[0][3], best_lists[0][4], best_beta_list[0]))
        print('\nbest recall: {:.5g} {:.5g} {:.5g} {:.5g} {:.5g}, beta: {}'.
              format(best_lists[1][0], best_lists[1][1], best_lists[1][2], best_lists[1][3], best_lists[1][4], best_beta_list[1]))
        print('\nbest hr: {:.5g} {:.5g} {:.5g} {:.5g} {:.5g}, beta: {}'.
              format(best_lists[2][0], best_lists[2][1], best_lists[2][2], best_lists[2][3], best_lists[2][4], best_beta_list[2]))
        print('\nbest mrr: {:.5g} {:.5g} {:.5g} {:.5g} {:.5g}, beta: {}'.
              format(best_lists[3][0], best_lists[3][1], best_lists[3][2], best_lists[3][3], best_lists[3][4], best_beta_list[3]))
        print('\nbest ndcg: {:.5g} {:.5g} {:.5g} {:.5g} {:.5g}, beta: {}'.
              format(best_lists[4][0], best_lists[4][1], best_lists[4][2], best_lists[4][3], best_lists[4][4], best_beta_list[4]))
        # '''



# pop_result_path = '../Pop_predict_cameraready/result/'
# model_result_path = '../Pop_predict_baseline_output/'
pop_result_path = '/Users/jingjiazheng/Project/data/cam_temp_store/result/'
model_result_path = '/Users/jingjiazheng/Project/data/cam_temp_store/'

if __name__ == '__main__':
    # pop_file = 'douban_movie_review.csv_gt_toppop.pop'
    # pop_file = 'reviews_Video_Games_5.json.gz_gt_toppop.pop'
    # pop_file = 'reviews_Home_and_Kitchen_5.json.gz_gt_toppop.pop'

    # pop_file = 'new_douban_movie_review_alpha_0.5_a_1.0_1.0_1.0_1.0_2023-08-136.pop'

    pop_file = 'new_reviews_Home_and_Kitchen_5_alpha_0.5_a_1.0_1.0_1.0_1.0_2023-08-13-184.pop'

    # pop_file = 'new_reviews_Video_Games_5_alpha_0.5_a_1.0_1.0_1.0_1.0_2023-08-13-20.pop'

    model_num = 7

    # model_result_file = 'Caser_test_douban_movie_review_2023-08-10-17.out'
    # model_result_file = 'HGN_test_douban_movie_review_2023-08-10-16.npy'
    # model_result_file = 'ICLRec_test_douban_movie_review_2023-08-12-1900485.npy'
    # model_result_file = 'NCF_douban_movie_review.out'
    # model_result_file = 'SASRec_test_douban_movie_review_2023-08-14-0.733970.out'
    # model_result_file = 'SHT_pred_douban_movie_review.out.npy'
    # model_result_file = 'STOSA_test_douban_movie_review_2023-08-12-0.189265.out.npy'
    # model_result_file = 'tra_method_douban_movie_review_tra_2023-08-14-014116.280772_3_model_result.npy'
    #
    # model_result_file = 'Caser_test_reviews_Video_Games_5_2023-08-10-17.out'
    # model_result_file = 'HGN_test_reviews_Video_Games_5_2023-08-10-11.npy'
    # model_result_file = 'ICLRec_test_reviews_Video_Games_5_2023-08-12-09.667582.npy'
    # model_result_file = 'NCF_reviews_Video_Games_5.out'
    model_result_file = 'SASRec_test_reviews_Video_Games_5_2023-08-14-116.141342.out'
    # model_result_file = 'SHT_pred_reviews_Video_Games_5.out.npy'
    # model_result_file = 'STOSA_test_reviews_Video_Games_5_2023-08-12-01.337000.out.npy'
    # model_result_file = 'tra_method_reviews_Video_Games_5_tra_2023-08-14-014142.314907_3_model_result.npy'
    #
    # model_result_file = 'Caser_test_reviews_Home_and_Kitchen_5_2023-08-10-17.out'
    # model_result_file = 'HGN_test_reviews_Home_and_Kitchen_5_2023-08-11.586991.npy'
    # model_result_file = 'ICLRec_test_reviews_Home_and_Kitchen_5_2023-08-1.016886.npy'
    # model_result_file = 'NCF_reviews_Home_and_Kitchen_5.out'
    # model_result_file = 'SASRec_test_reviews_Home_and_Kitchen_5_2023-08-12-0.387281.out'
    # model_result_file = 'SHT_pred_reviews_Home_and_Kitchen_5.out.npy'
    # model_result_file = 'STOSA_test_reviews_Home_and_Kitchen_5_2023-08-12-04.854654.out.npy'
    # model_result_file = 'tra_method_reviews_Home_and_Kitchen_5_tra_2023-08-10-163557.462386_8_model_result.npy'




    if 'STOSA' in model_result_file:
        model_type = 'STOSA'
        large_value_first = False
    elif 'tra' in model_result_file:
        model_type = 'traditional_methods'
        large_value_first = True
    elif 'ICLRec' in model_result_file:
        model_type = 'ICLRec'
        large_value_first = True
    elif 'Caser' in model_result_file:
        model_type = 'Caser'
        large_value_first = False
    elif 'HGN' in model_result_file:
        model_type = 'HGN'
        large_value_first = True
    elif 'SASRec' in model_result_file:
        model_type = 'SASRec'
        large_value_first = False
    elif 'NCF' in model_result_file:
        model_type = 'NCF'
        large_value_first = True
    elif 'SHT' in model_result_file:
        model_type = 'SHT'
        large_value_first = True

    if 'douban' in pop_file:
        dataset = 'douban_movie_review'
    elif 'Video' in pop_file:
        dataset = 'reviews_Video_Games_5'
    elif 'Home_and_Kitchen' in pop_file:
        dataset = 'reviews_Home_and_Kitchen_5'

    processed_path = '/Users/jingjiazheng/Project/data/cam_temp_store/tra_data/{}/'.format(dataset)
    test_tri_path = processed_path + '{}_tra_test.txt'.format(dataset)
    gt_file = test_tri_path

    beta = None
    print('pop_file: {}'.format(pop_file))
    print('model_result_file: {}'.format(model_result_file))
    precess_result_pop(dataset, model_type, large_value_first, pop_file, model_result_file, gt_file, beta, model_num)

