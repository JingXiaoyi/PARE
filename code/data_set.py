import math
import os
import gzip
import json
import sys
import time
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.utils.data as data

import Config

random.seed(2023)

generate_traditional_set = False  # If generate dataset for traditional model

from pypinyin import lazy_pinyin
from matplotlib import pyplot as plt


# ''.join(lazy_pinyin(dataset)
def sta_dataset(dict_item_time, dict_side_info, dict_genre):
    item_in_each_genre_month = [[0 for _ in range(12)] for _ in range(len(dict_genre))]
    num_genre = [0 for _ in range(len(dict_genre))]

    for item, inters in dict_item_time.items():
        side_info = eval(dict_side_info[item])
        for t in inters:
            for genre in side_info[0]:
                num_genre[genre] += 1
                item_in_each_genre_month[genre][t % 12] += 1

    for key in dict_genre:
        print(key, dict_genre[key], num_genre[dict_genre[key]])
        print(item_in_each_genre_month[dict_genre[key]])

        # draw graph
        genre_pinyin = ''.join(lazy_pinyin(key))
        plt.title('{}-total: {}'.format(genre_pinyin, num_genre[dict_genre[key]]))
        plt.xlabel('month')
        plt.ylabel('inters in each month')
        month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
        plt.bar(month, item_in_each_genre_month[dict_genre[key]])
        plt.grid()
        plt.savefig('./dataset_statistic_fig/{}.png'.format('monthly_inters_' + genre_pinyin))
        plt.close()
    exit()


def sta_top_pop(config:Config.Config, dict_item_pop, dict_item_idx, test_start_time):
    cut_off = [1, 3, 6, 12, 5, 10, 20]
    list_item_in_cutoff_range = np.zeros((len(dict_item_idx), len(cut_off)+1))
    for key in tqdm(dict_item_pop.keys()):
        item_pop = dict_item_pop[key]
        if key in dict_item_idx:
            idx = dict_item_idx[key]

            for i in range(len(cut_off)):
                # print(len(item_pop))
                # print(test_start_time)
                for pop in item_pop[test_start_time - cut_off[i] : test_start_time]:
                    list_item_in_cutoff_range[idx][i] += pop

            list_item_in_cutoff_range[idx][-1] += item_pop[test_start_time]
    path = './result/'+config.dataset+'_{}_toppop.pop'
    for cut_off_num in range(len(cut_off)):
        with open(path.format(cut_off[cut_off_num]), 'w') as f:
            for i in range(len(list_item_in_cutoff_range)):
                f.write('{} {}\n'.format(i, list_item_in_cutoff_range[i][cut_off_num]))
    with open(path.format('gt'), 'w') as f:
        for i in range(len(list_item_in_cutoff_range)):
            f.write('{} {}\n'.format(i, list_item_in_cutoff_range[i][-1]))


    # for key in tqdm(dict_item_time.keys()):
    #     unit_times = dict_item_time[key]
    #     if key in dict_item_idx:
    #         idx = dict_item_idx[key]
    #         for i in unit_times:
    #             for j in range(len(cut_off)):
    #                 if test_start_time-cut_off[j] <= i < test_start_time:
    #                     list_item_in_cutoff_range[idx][j] += 1
    #             if i == test_start_time:
    #                 list_item_in_cutoff_range[idx][4] += 1
    # path = './result/'+config.dataset+'_{}_toppop.pop'
    # for cut_off_num in range(len(cut_off)):
    #     with open(path.format(cut_off[cut_off_num]), 'w') as f:
    #         for i in range(len(list_item_in_cutoff_range)):
    #             f.write('{} {}\n'.format(i, list_item_in_cutoff_range[i][cut_off_num]))
    # with open(path.format('gt'), 'w') as f:
    #     for i in range(len(list_item_in_cutoff_range)):
    #         f.write('{} {}\n'.format(i, list_item_in_cutoff_range[i][-1]))


def sta_dataset_x_time_y_mean_std(config:Config.Config, dict_item_time):
    max_time_all = 0
    dict_num_in_each_month = {}
    for key in tqdm(dict_item_time.keys()):
        unit_times = dict_item_time[key]
        min_time = min(unit_times)
        unit_times = [i-min_time for i in unit_times]
        max_time = max(unit_times)
        # unit_times = [unit_times.count(i) for i in range(max_time+

        for i in range(max_time + 1):
            if not i in dict_num_in_each_month:
                dict_num_in_each_month[i] = []
            dict_num_in_each_month[i].append(unit_times.count(i))
        max_time_all = max(max_time, max_time_all)

    mean_all, std_all, amount_all = [], [], []
    for i in range(max_time_all+1):
        mean_all.append(np.mean(dict_num_in_each_month[i]))
        std_all.append(np.std(dict_num_in_each_month[i]))
        amount_all.append(np.sum(dict_num_in_each_month[i]))

    fig_x = list(range(max_time_all+1))
    # plt.title('{}-total: {}'.format(config.dataset, np.sum(amount_all)))
    # plt.xlabel('time')
    # plt.ylabel('amount')
    # plt.plot(fig_x, amount_all)
    # plt.grid()
    # plt.savefig('./dataset_statistic_fig/{}.png'.format(config.dataset + '_amount'))
    # plt.close()

    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(111)
    line_mean = ax1.plot(fig_x, mean_all, '-', label='mean', color='cornflowerblue')
    ax2 = ax1.twinx()
    line_std = ax2.plot(fig_x, std_all, '-', label='std', color='orange')

    lines = line_mean + line_std
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs)
    ax1.grid()
    ax1.set_title('dataset statistic ({})'.format(config.dataset))
    ax1.set_xlabel('Month')
    ax1.set_ylabel('mean')
    ax2.set_ylabel('std')
    ax1.set_xlim(xmin=0)
    ax1.set_ylim(ymin=0)
    ax2.set_xlim(xmin=0)
    ax2.set_ylim(ymin=0)
    # ax2.set_ylim(bottom=0.)
    # ax2.set_ylim(bottom=0.)
    plt.savefig('./dataset_statistic_fig/{}.pdf'.format(config.dataset + '_meanstd'))
    plt.close()

    # plt.title('{}-total: {}'.format(config.dataset, np.sum(amount_all)))
    # plt.xlabel('time')
    # plt.ylabel('mean')
    # plt.plot(fig_x, mean_all)
    # plt.grid()
    # plt.savefig('./dataset_statistic_fig/{}.png'.format(config.dataset + '_mean'))
    # plt.close()
    #
    # plt.title('{}-total: {}'.format(config.dataset, np.sum(amount_all)))
    # plt.xlabel('time')
    # plt.ylabel('std')
    # plt.plot(fig_x, std_all)
    # plt.grid()
    # plt.savefig('./dataset_statistic_fig/{}.png'.format(config.dataset + '_std'))
    # plt.close()
    exit()


def get_dict_a1_a2(df, json_root, dataset_name, a1, a2, config):
    dict_a1_a2 = {}
    print('num_{}: {}'.format(a1, len(df[a1].unique())))
    json_path = '{}{}_dict_{}_{}.json'.format(json_root, dataset_name.split('.')[0], a1, a2)

    if os.path.exists(json_path):
        print('dict_{}_{} json found'.format(a1, a2))
        inter_json = json.load(open(json_path, 'r'))
        dict_a1_a2 = inter_json['dict_{}_{}'.format(a1, a2)]
    else:
        print('dict_{}_{} json not found'.format(a1, a2))
        for _, row in df.iterrows():
            if row[a1] not in dict_a1_a2:
                dict_a1_a2[row[a1]] = [row[a2]]
            else:
                dict_a1_a2[row[a1]].append(row[a2])
        print(json_path)
        json.dump({'dict_{}_{}'.format(a1, a2): dict_a1_a2}, open(json_path, 'w'))

    return dict_a1_a2


def filter_ui_limit(l, user_limit, item_limit, user_index, item_index):
    ['item', 'user', 'rate', 'time']
    dict_user, dict_item, ok_user, ok_item = {}, {}, {}, {}
    new_list = []
    num_bad_user, num_bad_item = 0, 0
    for line in l:
        user = line[user_index]
        item = line[item_index]
        dict_user[user] = dict_user[user] + 1 if user in dict_user else 1
        dict_item[item] = dict_item[item] + 1 if item in dict_item else 1
    for key, value in dict_user.items():
        ok_user[key] = True if value >= user_limit else False
        if value < user_limit:
            num_bad_user += 1
    for key, value in dict_item.items():
        ok_item[key] = True if value >= item_limit else False
        if value < item_limit:
            num_bad_item += 1

    for line in l:
        user = line[user_index]
        item = line[item_index]
        if ok_user[user] and ok_item[item]:
            new_list.append(line)
    print('num_user: {}, num_item: {}, num_bad_user: {}, num_bad_item: {}, len_new_list: {}'.
          format(len(dict_user), len(dict_item), num_bad_user, num_bad_item, len(new_list)))
    if num_bad_user > 0 or num_bad_item > 0:
        return filter_ui_limit(new_list, user_limit, item_limit, user_index, item_index)
    else:
        return new_list


def filter_user_item_limit(df, user_limit, item_limit):
    df_columns = df.columns.values.tolist()
    user_index = df_columns.index('user')
    item_index = df_columns.index('item')
    print('user_index: {}, item_index: {}'.format(user_index, item_index))
    print(df_columns)
    data_list = df.values.tolist()
    del df
    data_filtered = filter_ui_limit(data_list, user_limit, item_limit, user_index, item_index)
    df = pd.DataFrame(data_filtered, columns=df_columns)
    return df


def list_to_csv(data_list, csv_columns):
    df = {}
    for i in range(len(csv_columns)):
        df[csv_columns[i]] = [line[i] for line in data_list]
    df = pd.DataFrame(df)
    return df


def make_dataset(config: Config.Config):
    main_path = config.main_path
    processed_path = config.processed_path
    train_path = config.train_path
    valid_path = config.valid_path
    test_path = config.test_path
    dataset = config.dataset
    time_unit = config.time_unit
    pop_time_unit = config.pop_time_unit
    pop_history_length = config.pop_history_length
    pos_item_pop_limit = config.pos_item_pop_limit
    neg_item_num = config.neg_item_num

    douban_rate_limit = 3  # interactions with rate less than the number is regarded as negative inter

    do_sta = False

    if os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path) and not do_sta:
        return

    ############################################ read dataset ##################################################
    if do_sta:
        print('\n Conducting dataset statistic...\n')
    print('Processed dataset not found. Processing begin:')
    if not os.path.exists(main_path + dataset):
        print('Dataset not found! Path: {}'.format(main_path + dataset))
        exit()
    if 'Amazon' in config.main_path:
        g = gzip.open(main_path + dataset, 'rb')
        df, idx = {}, 0
        for line in g:
            df[idx] = eval(line)
            idx += 1
        df = pd.DataFrame.from_dict(df, orient='index', dtype=str) \
            .drop(columns=['reviewerName', 'helpful', 'reviewText', 'summary', 'reviewTime']) \
            .rename(columns={'reviewerID': 'user', 'asin': 'item', 'unixReviewTime': 'time', 'overall': 'rate'})
        print('len_df: {}'.format(len(df)))
    else:
        # ['Bad', 'Reply', 'MovieID', 'UserID', 'ReviewID', 'Rate', 'Time', 'Good']
        df = pd.read_csv(main_path + dataset, index_col=0, header=0, dtype=str) \
            .drop(columns=['Good', 'Bad', 'Reply', 'ReviewID']) \
            .rename(columns={'UserID': 'user', 'MovieID': 'item', 'Time': 'time', 'Rate': 'rate'})
        df_columns = df.columns.values.tolist()
        data_list = df.values.tolist()
        del df

        data_filter_rate = []
        print(df_columns)
        rate_position = df_columns.index('rate')
        for line in data_list:
            if float(line[rate_position]) >= douban_rate_limit:
                data_filter_rate.append(line)
        df = pd.DataFrame(data_filter_rate, columns=df_columns)
        print('Original Length: {}, Length after filter rate: {}'.format(len(data_list), len(data_filter_rate)))

    # drop duplicate
    df.drop_duplicates(subset=['user', 'item'], keep='first', inplace=True)
    df.reset_index(inplace=True)

    # filter user_limit and item_limit
    df = filter_user_item_limit(df=df, user_limit=config.user_limit, item_limit=config.item_limit)

    # Find first inter time of item as release time
    list_time = list(map(int, map(float, df['time'])))
    df['time'] = list_time  # convert to int
    min_time_all = min(list_time)
    list_time = [int((i - min_time_all) / (time_unit * pop_time_unit)) for i in list_time]
    df['unit_time'] = list_time

    max_time = 0
    dict_item_time = get_dict_a1_a2(df, processed_path, dataset, 'item', 'unit_time', config)
    for key in list(dict_item_time.keys()):
        values = dict_item_time[key]
        if max(values) - min(values) < 2 * config.test_time_range + 1:
            del dict_item_time[key]
        else:
            max_time = max(max_time, max(values))

    dict_item_time_release = {}
    for key in dict_item_time:
        dict_item_time_release[key] = int(min(dict_item_time[key]))

    ########################################### Statistic Item pop ##################################################

    dict_item_pop = {}
    for item, item_time in dict_item_time.items():
        dict_item_pop[item] = [item_time.count(i) for i in range(max_time + 1)]

    ###################################### Process Side information ##################################################

    if config.is_douban:
        df_side_info = pd.read_csv(config.side_info_path, header=0, dtype=str)
        df_side_info = df_side_info.drop(columns=['Year', 'Screenwriter', 'Website', 'District', 'Language',
                                                  'Release_Date', 'Also_Called', 'Length', 'Alternate_name', 'IMDb',
                                                  'Description', 'Poster', 'Rate', 'Rate_Num',
                                                  'Ratio5', 'Ratio4', 'Ratio3', 'Ratio2', 'Ratio1', 'MovieName']) \
            .rename(columns={'Director': 'director', 'Actor': 'actor', 'Genre': 'genre', 'MovieID': 'name'})

    else:
        idx = 0
        dict_side_info = {}
        g = gzip.open(config.side_info_path, 'rb')
        for line in g:
            dict_side_info[idx] = eval(line)
            idx += 1
        df_side_info = pd.DataFrame.from_dict(dict_side_info, orient='index')

        df_side_info = df_side_info.drop(columns=['description', 'price', 'imUrl',
                                                  'related', 'salesRank', 'title', 'brand']) \
            .rename(columns={'categories': 'genre', 'asin': 'name'})
        print(len(df_side_info))
        print()

    # filter valid items

    # for i in df_side_info['name'][0:100]:
    #     print(i)
    #     if i in list(dict_item_time.keys()):
    #         print('yes')
    #     else:
    #         print('no')
    # exit()
    # print(len(dict_item_time))
    # print(dict_item_time)
    # print(len(df_side_info))
    # drop duplicate
    df_side_info = df_side_info[df_side_info['name'].isin(dict_item_time)]
    df_side_info.drop_duplicates(subset=['name'], keep='first', inplace=True)
    df_side_info.reset_index(inplace=True)
    if len(df_side_info) != len(dict_item_time):
        print('len(df_side_info): {}'.format(len(df_side_info)))
        print('len(dict_item_time): {}'.format(len(dict_item_time)))
        print('find item without side information or other error')
        # exit()
    dict_side_info = {}
    dict_director, dict_actor, dict_genre = {'padding': 0}, {'padding': 0}, {'padding': 0}
    num_director, num_actor, num_genre, max_genre = 1, 1, 1, 0
    if config.is_douban:
        for i in range(len(df_side_info)):
            director = eval(df_side_info['director'][i])
            actor = eval(df_side_info['actor'][i])
            genre = eval(df_side_info['genre'][i])

            director = [director[0]] if len(director) > 0 else ['padding']
            max_genre = max(max_genre, len(genre))
            if len(genre) == 0:
                genre = ['padding']
            if len(actor) == 0:
                actor = ['padding']
            if len(actor) >= 5:
                actor = actor[0:5]
            else:
                actor = actor + ['padding' for _ in range(5 - len(actor))]

            for g in range(len(genre)):
                if genre[g] not in dict_genre:
                    dict_genre[genre[g]] = num_genre
                    num_genre += 1
                genre[g] = dict_genre[genre[g]]

            if director[0] not in dict_director:
                dict_director[director[0]] = num_director
                num_director += 1
            director[0] = dict_director[director[0]]

            for j in range(len(actor)):
                if actor[j] not in dict_actor:
                    dict_actor[actor[j]] = num_actor
                    num_actor += 1
                actor[j] = dict_actor[actor[j]]
            dict_side_info[df_side_info['name'][i]] = [genre, director, actor]
        for key in dict_side_info:
            dict_side_info[key][0] = dict_side_info[key][0] + [0 for _ in range(max_genre-len(dict_side_info[key][0]))]
            dict_side_info[key] = str(dict_side_info[key])
    else:
        for i in range(len(df_side_info)):
            genre = df_side_info['genre'][i]

            genre = [genre[0]] if len(genre) > 0 else ['padding']
            if len(genre[0]) > 3:
                genre[0] = genre[0][0:3]
            genre = ['_'.join(genre[0])]

            if genre[0] not in dict_genre:
                dict_genre[genre[0]] = num_genre
                num_genre += 1
            genre[0] = dict_genre[genre[0]]

            dict_side_info[df_side_info['name'][i]] = str([genre])

    print('num_genre: {}, num_actor: {}, num_director: {}'.format(num_genre, num_actor, num_director))

    ###################################### generate train/valid/test set ##############################################

    train_dataset, valid_dataset, test_dataset = [], [], []
    pos_valid, pos_test = {}, {}
    dict_item_idx = {}
    num_item = 0

    valid_start_time = max_time - 2 * config.test_time_range + 1
    test_start_time = max_time - config.test_time_range + 1
    valid_list = list(range(valid_start_time, test_start_time))
    test_list = list(range(test_start_time, max_time + 1))

    ################## statistic dataset ####################
    # sta_dataset(dict_item_time, dict_side_info, dict_genre)
    # sta_dataset_x_time_y_mean_std(config, dict_item_time)
    #########################################################

    for item, item_pop in dict_item_pop.items():
        item_time_release = dict_item_time_release[item]
        if item not in dict_side_info:
            dict_side_info[item] = '[[0]]'
        side_info = dict_side_info[item]
        pos_list, neg_list = [], []

        if item_time_release >= valid_start_time:
            print(max_time)
            print(valid_start_time)
            print(dict_item_time_release[item])
            print(dict_item_time[item])
            print(dict_item_pop[item])
            print('ERROR: item_time_release>=valid_start_time')
            # exit()

        # sample neg time
        for t in range(valid_start_time):
            if item_pop[t] > pos_item_pop_limit:
                pos_list.append(t)

        # construct train set
        if len(pos_list) > 0:
            if item not in dict_item_idx:
                dict_item_idx[item] = num_item
                num_item += 1
        for time_now in pos_list:
            pop_history = item_pop[dict_item_time_release[item]: time_now]
            if len(pop_history) == 0:
                pop_history = [0]
            valid_pop_len = len(pop_history)
            pop_history += [-1 for _ in range(max_time-valid_pop_len + 1)]
            pop_gt = item_pop[time_now]
            line = [dict_item_idx[item], item_time_release, side_info, time_now, str(pop_history), pop_gt, valid_pop_len]
            train_dataset.append(line)

        # construct valid set
        if max([item_pop[i] for i in valid_list]) > 0 and item in dict_item_idx:
            for time_now in valid_list:
                pop_history = item_pop[dict_item_time_release[item]: time_now]
                if len(pop_history) == 0:
                    pop_history = [0]
                valid_pop_len = len(pop_history)
                pop_history += [-1 for _ in range(max_time-valid_pop_len + 1)]
                pop_gt = item_pop[time_now]
                line = [dict_item_idx[item], item_time_release, side_info, time_now, str(pop_history), pop_gt, valid_pop_len]
                pos_valid[item] = True
                valid_dataset.append(line)

        # construct test set
        if max([item_pop[i] for i in test_list]) > 0 and item in dict_item_idx:
            for time_now in test_list:
                pop_history = item_pop[dict_item_time_release[item]: time_now]
                if len(pop_history) == 0:
                    pop_history = [0]
                valid_pop_len = len(pop_history)
                pop_history += [-1 for _ in range(max_time-valid_pop_len + 1)]
                pop_gt = item_pop[time_now]
                line = [dict_item_idx[item], item_time_release, side_info, time_now, str(pop_history), pop_gt, valid_pop_len]
                pos_test[item] = True
                test_dataset.append(line)

    sta_top_pop(config, dict_item_pop, dict_item_idx, test_start_time)
    #########################################################

    train_df = list_to_csv(train_dataset, ['item', 'time_release', 'side_info', 'time', 'pop_history', 'pop_gt', 'valid_pop_len'])
    valid_df = list_to_csv(valid_dataset, ['item', 'time_release', 'side_info', 'time', 'pop_history', 'pop_gt', 'valid_pop_len'])
    test_df = list_to_csv(test_dataset, ['item', 'time_release', 'side_info', 'time', 'pop_history', 'pop_gt', 'valid_pop_len'])

    train_df.to_csv(train_path)
    print('Train set saved len_csv: {}, item_num: {}, num_zero_pop: {}'
          .format(len(train_df), len(train_df['item'].unique()), list(train_df['pop_gt']).count(0)))
    print('num_inter: {}'.format(np.sum(list(train_df['pop_gt']))))
    valid_df.to_csv(valid_path)
    print('Valid set saved len_csv: {}, item_num: {}, num_zero_pop: {}'
          .format(len(valid_df), len(valid_df['item'].unique()), list(valid_df['pop_gt']).count(0)))
    print('num_inter: {}'.format(np.sum(list(valid_df['pop_gt']))))
    test_df.to_csv(test_path)
    print('Test set saved len_csv: {}, item_num: {}, num_zero_pop: {}'
          .format(len(test_df), len(test_df['item'].unique()), list(test_df['pop_gt']).count(0)))
    print('num_inter: {}'.format(np.sum(list(test_df['pop_gt']))))

    ######################################### save dataset info #################################################

    with open(config.info_path, 'w') as f:
        # num_user num_item max_pop max_time list_item_time list_item_time_release
        f.write('{}\n'.format(str(num_item)))
        f.write('{}\n'.format(str(max_time)))
        f.write('{}\n'.format(str(dict_item_idx)))
        f.write('{}\n'.format(str(dict_item_pop)))
        f.write('{}\n'.format(str(dict_item_time_release)))
        f.write('{}\n'.format(str([num_genre, num_director, num_actor])))
        f.write('{}\n'.format(str(dict_side_info)))

    # generate_tra_set(config, df, max_time, dict_item_idx)
    return


def generate_tra_set(config: Config.Config, df, max_time, dict_item_idx):
    config.test_time_range
    valid_start_time = max_time - 2 * config.test_time_range + 1
    test_start_time = max_time - config.test_time_range + 1
    valid_range = list(range(valid_start_time, test_start_time))
    test_range = list(range(test_start_time, max_time + 1))

    df = df[df['item'].isin(dict_item_idx)]

    list_train, list_valid, list_test = [], [], []
    dict_user_idx = {}
    num_user = 0
    # generate train set, filter trained user to ensure every user occur in train set
    for user, item, t, rate, unit_time in zip(df['user'], df['item'], df['time'], df['rate'], df['unit_time']):
        if unit_time not in valid_range and unit_time not in test_range:
            if user not in dict_user_idx:
                dict_user_idx[user] = num_user
                num_user += 1
            line = [dict_user_idx[user], dict_item_idx[item], rate, t]
            list_train.append(line)
    # generate valid and test set
    for user, item, t, rate, unit_time in zip(df['user'], df['item'], df['time'], df['rate'], df['unit_time']):
        if user in dict_user_idx:
            line = [dict_user_idx[user], dict_item_idx[item], rate, t]
            if unit_time in valid_range:
                list_valid.append(line)
            elif unit_time in test_range:
                list_test.append(line)

    # save dataset
    train_tri_path = config.processed_path + '{}_tra_train.txt'.format(config.dataset.split('.')[0])
    valid_tri_path = config.processed_path + '{}_tra_validate.txt'.format(config.dataset.split('.')[0])
    test_tri_path = config.processed_path + '{}_tra_test.txt'.format(config.dataset.split('.')[0])

    with open(train_tri_path, 'w') as f:
        for line in list_train:
            f.write('{} {} {} {}\n'.format(line[0], line[1], line[2], line[3]))
    with open(valid_tri_path, 'w') as f:
        for line in list_valid:
            f.write('{} {} {} {}\n'.format(line[0], line[1], line[2], line[3]))
    with open(test_tri_path, 'w') as f:
        for line in list_test:
            f.write('{} {} {} {}\n'.format(line[0], line[1], line[2], line[3]))
    print('Train tra set done, len: {}, num_user: {}, num_item: {}'
          .format(len(list_train), len(set([i[0] for i in list_train])), len(set([i[1] for i in list_train]))))
    print('Valid tra set done, len: {}, num_user: {}, num_item: {}'
          .format(len(list_valid), len(set([i[0] for i in list_valid])), len(set([i[1] for i in list_valid]))))
    print('Test tra set done, len: {}, num_user: {}, num_item: {}'
          .format(len(list_test), len(set([i[0] for i in list_test])), len(set([i[1] for i in list_test]))))
    return


def loaded_json(config):
    if not config.json_loaded:
        with open(config.info_path, 'r') as f:
            lines = f.readlines()
            # num_item, max_time, dict_item_idx, dict_item_pop, dict_item_time_release, num_side_info, dict_side_info
            config.num_item = int(lines[0].strip())
            config.max_time = int(lines[1].strip())
            config.dict_item_idx = eval(lines[2].strip())
            config.dict_item_pop = eval(lines[3].strip())
            config.dict_item_time_release = eval(lines[4].strip())
            config.num_side_info = eval(lines[5].strip())
            config.dict_side_info = eval(lines[6].strip())

            # generate highest_pop groundtruth
            # for i in range(len(config.list_item_time)):
            #     min_time = min(config.list_item_time[i])
            #     unit_time_list = [(j-min_time) // config.pop_time_unit for j in config.list_item_time[i]]
            #     config.gt_highest_pop.append(unit_time_list.count(max(set(unit_time_list), key=unit_time_list.count)))

        config.json_loaded = True


def pop_func(x):
    if x>0:
        return math.log(x + 1)
    else:
        return x
    # return x


class Data(data.Dataset):
    def __init__(self, config:Config.Config, set_type):
        super(Data, self).__init__()
        self.config = config
        self.set_type = set_type
        make_dataset(config)
        loaded_json(config)
        self.data = self.load_dataset()
        self.process_pop()

        self.data_ori = self.data

    def process_pop(self):
        self.data['pop_gt'] = list(map(pop_func, self.data['pop_gt']))
        self.data['pop_history'] = [list(map(pop_func, i)) for i in self.data['pop_history']]

    def load_dataset(self):
        if self.set_type == 'Train':
            data_path = self.config.train_path
        elif self.set_type == 'Valid':
            data_path = self.config.valid_path
        elif self.set_type == 'Test':
            data_path = self.config.test_path
        else:
            print('Dataset type error!')
            exit()
        df = pd.read_csv(data_path, header=0, index_col=0,
                         dtype={'item': int, 'time_release': int, 'side_info': str,
                                'time': int, 'pop_history': str, 'pop_gt': int, 'valid_pop_len': int})

        df['side_info'] = list(map(eval, df['side_info']))
        df['pop_history'] = list(map(eval, df['pop_history']))
        return df
    def sample_neg(self):
        # print('sampling neg')
        if len(self.config.dict_idx_item) == 0:
            print('processing dict_idx_item')
            for key in self.config.dict_item_idx.keys():
                self.config.dict_idx_item[self.config.dict_item_idx[key]] = key

        neg_all = []
        for i in range(len(self.data_ori)):
            item_idx = self.data_ori['item'][i]
            item_name = self.config.dict_idx_item[item_idx]
            item_time_release = self.config.dict_item_time_release[item_name]

            item_pop = self.config.dict_item_pop[item_name][: -2*self.config.test_time_range]
            if self.config.max_time + 1 - 2 * self.config.test_time_range >= item_time_release:

                time_limit = 10
                neg_time = random.randint(item_time_release, self.config.max_time - 2 * self.config.test_time_range)
                while item_pop[neg_time] > self.config.pos_item_pop_limit and time_limit > 0:
                    time_limit -= 1
                    neg_time = random.randint(item_time_release, self.config.max_time - 2 * self.config.test_time_range)

                pop_history = item_pop[item_time_release: neg_time]
                if len(pop_history) == 0:
                    pop_history = [0]
                valid_pop_len = len(pop_history)
                pop_history += [-1 for _ in range(self.config.max_time - valid_pop_len + 1)]

                pop_gt = item_pop[neg_time]
                side_info = eval(self.config.dict_side_info[item_name])
                line = [item_idx, item_time_release, side_info, neg_time, pop_history, pop_gt, valid_pop_len]
                neg_all.append(line)

                # print('item = {}'.format(item_idx))
                # print('item = {}'.format(item.shape))
                # print('time_release = {}'.format(item_time_release))
                # print('time_release = {}'.format(time_release.shape))
                # print('item_genre = {}'.format(item_genre))
                # print('item_genre = {}'.format(item_genre.shape))
                # print('item_director = {}'.format(item_director))
                # print('item_director = {}'.format(item_director.shape))
                # print('item_actor = {}'.format(item_actor))
                # print('item_actor = {}'.format(item_actor.shape))
                # print('time = {}'.format(time))
                # print('neg_time = {}'.format(neg_time))
                # print('time = {}'.format(time.shape))
                # print('pop_history = {}'.format(pop_history))
                # print('item_pop = {}'.format(item_pop))
                # print('pop_history = {}'.format(pop_history.shape))
                # print('pop_gt = {}'.format(pop_gt))
                # print('pop_gt = {}'.format(pop_gt.shape))
                # print('valid_pop_len = {}'.format(valid_pop_len))
                # print('valid_pop_len = {}'.format(valid_pop_len.shape))
                # input('debug: neg_sample')

        # print(len(self.data_ori))
        # print(len(neg_all))
        data_new = pd.DataFrame({'item': [i[0] for i in neg_all],
                                 'time_release': [i[1] for i in neg_all],
                                 'side_info': [i[2] for i in neg_all],
                                 'time': [i[3] for i in neg_all],
                                 'pop_history': [i[4] for i in neg_all],
                                 'pop_gt': [i[5] for i in neg_all],
                                 'valid_pop_len': [i[6] for i in neg_all]})

        data_new['pop_gt'] = list(map(pop_func, data_new['pop_gt']))
        data_new['pop_history'] = [list(map(pop_func, i)) for i in data_new['pop_history']]
        self.data = pd.concat([self.data_ori, data_new], axis=0, ignore_index=True)
        # print('sample done, len_data: {}'.format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # ['item', 'time_release', 'side_info', 'time', 'pop_history', 'pop_gt', 'valid_pop_len']
        return [self.data['item'][idx], self.data['time_release'][idx], self.data['side_info'][idx],
                self.data['time'][idx], self.data['pop_history'][idx], self.data['pop_gt'][idx],
                self.data['valid_pop_len'][idx]]
'''
 Conducting dataset statistic...

Processed dataset not found. Processing begin:
['item', 'user', 'rate', 'time']
Original Length: 393100, Length after filter rate: 352336
user_index: 2, item_index: 1
['index', 'item', 'user', 'rate', 'time']
num_user: 33705, num_item: 7253, num_bad_user: 0, num_bad_item: 4191, len_new_list: 336924
num_user: 33689, num_item: 3062, num_bad_user: 0, num_bad_item: 0, len_new_list: 336924
num_item: 3062
dict_item_unit_time json found
3020
3511
num_genre: 29, num_actor: 7027, num_director: 1889
100%|████████████████████████████████████| 3020/3020 [00:00<00:00, 34615.07it/s]
Train set saved len_csv: 18737, item_num: 2795, num_zero_pop: 0
Valid set saved len_csv: 635, item_num: 635, num_zero_pop: 0
Test set saved len_csv: 302, item_num: 302, num_zero_pop: 0
Train tra set done, len: 329380, num_user: 33635, num_item: 2795
Valid tra set done, len: 1801, num_user: 1096, num_item: 590
Test tra set done, len: 497, num_user: 356, num_item: 273

 Conducting dataset statistic...

Processed dataset not found. Processing begin:
len_df: 231780
user_index: 1, item_index: 2
['index', 'user', 'item', 'rate', 'time']
num_user: 24303, num_item: 10672, num_bad_user: 0, num_bad_item: 4583, len_new_list: 201737
num_user: 24293, num_item: 6089, num_bad_user: 0, num_bad_item: 0, len_new_list: 201737
num_item: 6089
dict_item_unit_time json found
50953
6041
6041
num_genre: 122, num_actor: 0, num_director: 0
100%|████████████████████████████████████| 6041/6041 [00:00<00:00, 47220.20it/s]
Train set saved len_csv: 16479, item_num: 4211, num_zero_pop: 0
Valid set saved len_csv: 1116, item_num: 1116, num_zero_pop: 0
Test set saved len_csv: 1093, item_num: 1093, num_zero_pop: 0
Train tra set done, len: 169845, num_user: 23933, num_item: 4211
Valid tra set done, len: 1479, num_user: 875, num_item: 873
Test tra set done, len: 1268, num_user: 684, num_item: 789

 Conducting dataset statistic...

Processed dataset not found. Processing begin:
len_df: 1689188
user_index: 1, item_index: 2
['index', 'user', 'item', 'rate', 'time']
num_user: 192403, num_item: 63001, num_bad_user: 0, num_bad_item: 27927, len_new_list: 1507045
num_user: 192395, num_item: 35074, num_bad_user: 0, num_bad_item: 0, len_new_list: 1507045
num_item: 35074
dict_item_unit_time json found
50953
34799
86
len(df_side_info): 86
len(dict_item_time): 34799
find item without side information or other error
num_genre: 15, num_actor: 0, num_director: 0
100%|██████████████████████████████████| 34799/34799 [00:00<00:00, 46712.69it/s]
Train set saved len_csv: 137523, item_num: 24119, num_zero_pop: 0
Valid set saved len_csv: 10918, item_num: 10918, num_zero_pop: 0
Test set saved len_csv: 1749, item_num: 1749, num_zero_pop: 0
Train tra set done, len: 1304690, num_user: 190849, num_item: 24119
Valid tra set done, len: 33999, num_user: 18106, num_item: 9890
Test tra set done, len: 2456, num_user: 1926, num_item: 1296

 Conducting dataset statistic...

Processed dataset not found. Processing begin:
len_df: 551682
user_index: 1, item_index: 2
['index', 'user', 'item', 'rate', 'time']
num_user: 66519, num_item: 28237, num_bad_user: 0, num_bad_item: 13695, len_new_list: 462551
num_user: 66496, num_item: 14542, num_bad_user: 0, num_bad_item: 0, len_new_list: 462551
num_item: 14542
dict_item_unit_time json found
50953

len(df_side_info): 0
len(dict_item_time): 14243
find item without side information or other error
num_genre: 1, num_actor: 0, num_director: 0
100%|██████████████████████████████████| 14243/14243 [00:00<00:00, 54302.49it/s]
Train set saved len_csv: 39472, item_num: 8633, num_zero_pop: 0
Valid set saved len_csv: 4915, item_num: 4915, num_zero_pop: 0
Test set saved len_csv: 277, item_num: 277, num_zero_pop: 0
Train tra set done, len: 361005, num_user: 65588, num_item: 8633
Valid tra set done, len: 10730, num_user: 6437, num_item: 4413
Test tra set done, len: 245, num_user: 230, num_item: 162
'''