import os
import time as tt
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn


import Model
import Config
import evaluate
import data_set
from torch.optim import AdamW
from torch.optim import SGD
from torch.optim import Adam

np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, default='',
                    help="data_path")

# 'reviews_Video_Games_5'
# 'reviews_Home_and_Kitchen_5'
# 'douban_movie_review'

# Dataset settings
# Training settings
parser.add_argument("--batch_size", type=int, default=128,
                    help="batch size for training")
parser.add_argument("--lr", type=float, default=0.1,
                    help="learning rate")
parser.add_argument("--dataset", type=str, default='reviews_Video_Games_5',
                    help="dataset file name")
parser.add_argument("--time_unit", type=int, default=86400,
                    help="smallest time unit for model training")
parser.add_argument("--pop_time_unit", type=int, default=30,
                    help="smallest time unit for item popularity statistic")
parser.add_argument("--num_genre_period", type=int, default=12,
                    help="predefined genre period")
parser.add_argument("--test_time_range", type=int, default=1,
                    help="time range of test and valid test")
parser.add_argument("--user_limit", type=int, default=0,
                    help="filter user with inters less than user_limit")
parser.add_argument("--item_limit", type=int, default=10,
                    help="filter item with inters less than item_limit")
parser.add_argument("--pop_history_length", type=int, default=0,
                    help="length of RNN input")
parser.add_argument("--pos_item_pop_limit", type=int, default=2,
                    help="popularity count for positive item")
parser.add_argument("--neg_item_num", type=int, default=2,
                    help="num of neg_item/pos_item")



# pop_pred = history_average*alpha + history_average*(1-alpha)*pop_history_embed.squeeze()
# ema[:, i] = beta * matrix[:, i] + (1 - beta) * ema[:, i - 1]

parser.add_argument("--alpha", type=float, default=12,
                    help="parameter for cutoff")
parser.add_argument("--beta", type=float, default=0.7,
                    help="parameter for balance of pop_history and time")

parser.add_argument("--a1", type=float, default=1.0,
                    help="parameter for balance of pop_history and time")
parser.add_argument("--a2", type=float, default=1.0,
                    help="parameter for time_output")
parser.add_argument("--a3", type=float, default=1.0,
                    help="parameter for sideinfo_output")
parser.add_argument("--a4", type=float, default=1.0,
                    help="parameter for periodic_output")


'''


best_hr: [0.14325843 0.17134831]
best_ndcg: [0.07450982 0.08014864]
'''


parser.add_argument("--embed_size", type=int, default=512,
                    help="embedding size for embedding vectors")
parser.add_argument("--loss", type=str, default='hinge',
                    help="loss function, options: hinge, log, square_square, square_exp")
parser.add_argument("--epochs", type=int, default=20,
                    help="training epochs")
parser.add_argument("--dropout", type=float, default=0,
                    help="dropout rate")
parser.add_argument("--top_k", type=list, default=[1, 5, 10, 20],
                    help="compute metrics@top_k")
parser.add_argument("--max_pop", type=int, default=100,
                    help="max_pop in model")
parser.add_argument("--num_eval_count", type=int, default=30,
                    help="eval times in each epoch")
parser.add_argument("--eval_limit", type=int, default=5000,
                    help="batch size in eval to save memory")
# Running settings
parser.add_argument("--device", type=str, default="cuda:0",
                    help="choose gpu or cpu to train model")
parser.add_argument("--load_model", type=int, default=0,
                    help="to load model or not (0: False, 1: True)")
parser.add_argument("--save_model", type=int, default=0,
                    help="to save model or not (0: False, 1: True)")

args = parser.parse_args()

# Load config
config = Config.Config(args=args)

# Load gpu

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# train_dataset = data_set.Data(set_type='Train', config=config)
# train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
# train_loader.dataset.sample_neg()
# train_loader.dataset.sample_neg()
# for label, user, recent_items, time, item, time_release, item_pop, recent_items_tr, side_info in train_loader:
#     recent_items = torch.stack(recent_items, 0).transpose(0, 1)
#     recent_items_tr = torch.stack(recent_items_tr, 0).transpose(0, 1)
#     item_genre = side_info[0][0]
#     if 'douban' in config.dataset:
#         item_director = side_info[1][0]
#         item_actor = torch.stack(side_info[2], 0).transpose(0, 1)
#
#     print('item_genre = {}'.format(item_genre))
#     print('item_genre = {}'.format(item_genre.shape))
#     print('item_director = {}'.format(item_director))
#     print('item_director = {}'.format(item_director.shape))
#     print('item_actor = {}'.format(item_actor))
#     print('item_actor = {}'.format(item_actor.shape))
#     input('debug')

################################################### Create Dataset ####################################################

print('Preparing Dataset...')
train_dataset = data_set.Data(config=config, set_type='Train')
valid_dataset = data_set.Data(config=config, set_type='Valid')
test_dataset = data_set.Data(config=config, set_type='Test')

train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=config.eval_limit)
test_loader = data.DataLoader(test_dataset, batch_size=config.eval_limit)

################################################### Create Model ####################################################

print('Creating Model...')
if config.load_model:
    if not os.path.exists('./saved_model/best_model_douban_movie_review.csv.pth'):
        print('model_path not found, load fail.')
    model = torch.load('./saved_model/best_model_douban_movie_review.csv.pth', map_location=torch.device('cpu'))
    train_loader.dataset.sample_neg()
    with torch.no_grad():
        te_PRECISION, te_RECALL, te_HR, te_MRR, te_NDCG, teloss, teploss, tealoss, tpop_acc = \
            evaluate.evaluate(model, test_loader, config)

        print("Test(load_model) PRECISION: {} \n RECALL: {} \n HR: {} \n MRR: {} \n NDCG: {}"
              .format(te_PRECISION, te_RECALL, te_HR, te_MRR, te_NDCG))
        print('Popularity accuracy: {}'.format(str(tpop_acc)))
        print('all(pred_pop, pop_value, pred_pop*pop_value), pos(..., ..., ...), neg(..., ..., ...)')
        print('Done')
    exit()
model = Model.PopPredict(True, config)
model.to(config.device)






HR, NDCG, test_mse, pred_test = evaluate.calculate_hr_ndcg(model, config)
print('HR@5: {}, HR@10: {}, NDCG@5: {}, NDCG@10: {}'
      .format(HR[0], HR[1], NDCG[0], NDCG[1]))

exit()







##################################################### Training ######################################################

print('Training begin...')
count, best_mse, best_mae, best_epoch = 0, 1000000, 1000000, -1

list_loss = []
best_pred_test, best_pop_gt_test = [], []
list_train_loss, list_test_loss = [], []
mae_final, mse_final, item_final, pop_gt_final = 0, 0, [], []
pred_final_1, pred_final_2, pred_final_3 = [], [], []
mse_for_pred_final = [1000000, 1000000, 1000000]
best_hr, best_ndcg = [0, 0], [0, 0]
best_test_mse = 1000

optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=0.001)
# all_iteration = len(train_dataset.data) * config.epochs
train_loader.dataset.sample_neg()
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=(len(train_dataset.data) * config.epochs), eta_min=0)
#
# parameters for loss figure
# count_figure = 0
# fig_name_pre = '{}{}-{}'.format(config.fig_path, datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f'), config.dataset)
for epoch in range(config.epochs):
    print('\n')
    list_loss = [[], [], [], [], [], []]
    start_time = tt.time()
    # train_loader.dataset.sample_neg()
    model.train()
    model.is_training = True
    # ['item', 'time_release', 'side_info', 'time', 'pop_history', 'pop_gt']
    for item, time_release, side_info, time, pop_history, pop_gt, valid_pop_len in tqdm(train_loader):
        item_genre = torch.stack(side_info[0], 0).transpose(0, 1)
        pop_history = torch.stack(pop_history, 0).transpose(0, 1)

        if config.alpha > 0:
            valid_pop_len = torch.clamp(valid_pop_len, max=config.alpha)

        item_director, item_actor = torch.Tensor([0]), torch.Tensor([0])
        if config.is_douban:
            item_director = side_info[1][0]
            item_actor = torch.stack(side_info[2], 0).transpose(0, 1)

        # print('item = {}'.format(item))
        # print('item = {}'.format(item.shape))
        # print('time_release = {}'.format(time_release))
        # print('time_release = {}'.format(time_release.shape))
        # print('item_genre = {}'.format(item_genre))
        # print('item_genre = {}'.format(item_genre.shape))
        # print('item_director = {}'.format(item_director))
        # print('item_director = {}'.format(item_director.shape))
        # print('item_actor = {}'.format(item_actor))
        # print('item_actor = {}'.format(item_actor.shape))
        # print('time = {}'.format(time))
        # print('time = {}'.format(time.shape))
        # print('pop_history = {}'.format(pop_history))
        # print('pop_history = {}'.format(pop_history.shape))
        # print('pop_gt = {}'.format(pop_gt))
        # print('pop_gt = {}'.format(pop_gt.shape))
        # print('valid_pop_len = {}'.format(valid_pop_len))
        # print('valid_pop_len = {}'.format(valid_pop_len.shape))
        # input('debug: model input')

        item = item.to(config.device)
        time_release = time_release.to(config.device)
        item_genre = item_genre.to(config.device)
        item_director = item_director.to(config.device)
        item_actor = item_actor.to(config.device)
        time = time.to(config.device)
        pop_history = pop_history.to(torch.float).to(config.device)
        pop_gt = pop_gt.to(device=config.device, dtype=torch.float)
        valid_pop_len = valid_pop_len.to(torch.float).to(config.device)

        model.zero_grad()
        pop_history_output, time_output, sideinfo_output, periodic_output, pred = model(
            item=item,
            time_release=time_release,
            item_genre=item_genre,
            item_director=item_director,
            item_actor=item_actor,
            time=time,
            pop_history=pop_history,
            pop_gt=pop_gt,
            valid_pop_len=valid_pop_len
        )

        # print('pred = {}'.format(pred))
        # input('debug')
        criteria = nn.MSELoss()
        loss_1 = criteria(pop_history_output.squeeze(), pop_gt)
        loss_2 = criteria(time_output.squeeze(), pop_gt)
        loss_3 = criteria(sideinfo_output.squeeze(), pop_gt)
        loss_4 = criteria(periodic_output.squeeze(), pop_gt)
        loss_all = criteria(pred.squeeze(), pop_gt)
        loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_all

        # print('pred:', pred.squeeze())
        # print('pop_gt:', pop_gt)
        # input('debug: pred and pop_gt')
        # loss = torch.mean(torch.abs(pred.squeeze()-pop_gt))
        list_loss[0].append(loss_1.item())
        list_loss[1].append(loss_2.item())
        list_loss[2].append(loss_3.item())
        list_loss[3].append(loss_4.item())
        list_loss[4].append(loss_all.item())
        list_loss[5].append(loss.item())
        loss.backward()
        optimizer.step()

        # scheduler.step()
        # print('{:.5f}\t{:.5f}\t{:.5f}'.format(loss.item(), pop_loss.item(), loss.item()))
        # pop_loss_list.append(pop_loss.cpu().detach().numpy().tolist())
        # writer.add_scalar('data/loss', loss.item(), count)
    # Evaluating
    model.eval()
    model.is_training = False
    with torch.no_grad():
        print('epoch: {}'.format(epoch))
        HR, NDCG, test_mse, pred_test = evaluate.calculate_hr_ndcg(model, config)
        # if HR[1] > best_hr[1] or (HR[1] == best_hr[1] and best_test_mse > test_mse):
        if NDCG[1] > best_ndcg[1]: #  or (NDCG[1] == best_ndcg[1] and best_test_mse > test_mse):
        # if best_test_mse > test_mse:
            print('best test found!')
            best_test_mse = test_mse
            best_ndcg = NDCG
            best_hr = HR
            best_epoch = epoch
            best_pred = pred_test
            with open('{}.pop'.format(config.result_path), 'w') as f_result:
                for i in range(len(best_pred)):
                    f_result.write('{} {}\n'.format(i, best_pred[i]))
            print('{}.pop'.format(config.result_path).split('/')[-1])
            print('writing done')

        print('HR@5: {}, HR@10: {}, NDCG@5: {}, NDCG@10: {}'
              .format(HR[0], HR[1], NDCG[0], NDCG[1]))
        print('test_mse: pop_history: {:.5f}, time: {:.5f} sideinfo: {:.5f} periodic: {:.5f} final: {:.5f}'
              .format(test_mse[0], test_mse[1], test_mse[2], test_mse[3], test_mse[4]))
        print('Train_loss: pop_history: {:.5f}, time: {:.5f}, sideinfo: {:.5f}, periodic: {:.5f}, final: {:.5f}, sum: {:.5f}'
              .format(np.mean(list_loss[0]), np.mean(list_loss[1]), np.mean(list_loss[2]),
                      np.mean(list_loss[3]), np.mean(list_loss[4]), np.mean(list_loss[5])))


        # mae, mse, item_test, pred_test, pop_gt_test = evaluate.evaluate_pop_predict(model, test_loader, config)
        #
        # elapsed_time = tt.time() - start_time
        # print('Epoch: {}, Time: {}'.format(epoch, tt.strftime("%H: %M: %S", tt.gmtime(elapsed_time))))
        # print('Train_loss: {:.5f}'.format(np.mean(list_loss)))
        # print("Test MAE: {:.5f} MSE: {:.5f}".format(mae, mse))
        # list_train_loss.append(np.mean(list_loss))
        # list_test_loss.append(mse)
        # if mse < best_mse:
        #     print('New best test! ')
        #     best_mse = mse
        #     best_mae = mae
        #     best_epoch = epoch
        #     best_pred_test = pred_test
        #     pop_gt_test_fig = np.round(np.array(pop_gt_test))
        #     best_pop_gt_test = pop_gt_test
        #     if config.save_model:
        #         if not os.path.exists('./saved_model'):
        #             os.mkdir('./saved_model')
        #         torch.save(model, config.model_path)

            ############################### save fig ###############################

            # pop_gt_test_fig = np.round(np.exp(np.array(pop_gt_test))) - 1
            # pred_test_fig = np.round(np.exp(np.array(pred_test))) - 1
            #
            # # pop_gt_test_fig = np.round(np.exp(np.array(pop_gt_test)) *5)/5
            # # pred_test_fig = np.round(np.exp(np.array(pred_test)) *5)/5
            #
            # list_gt_pred = [(pop_gt_test_fig[i], pred_test_fig[i]) for i in range(len(pop_gt_test_fig))]
            # z_gt_pred = [list_gt_pred.count((pop_gt_test_fig[i], pred_test_fig[i])) for i in range(len(pop_gt_test_fig))]
            #
            # plt.clf()
            # plt.figure(figsize=(6, 4.5))
            # plt.title('gap between model prediction and ground truth')
            # plt.xlabel('ground truth')
            # plt.ylabel('pred_pop')
            # # plt.scatter(pop_gt_test_fig, pred_test_fig, alpha=0.1, linewidths=0)
            #
            # plt.scatter(pop_gt_test_fig, pred_test_fig, c=z_gt_pred, linewidths=0, cmap="coolwarm", s=25)
            # plt.colorbar()
            #
            # # print(pred_test_fig)
            # # print(pop_gt_test_fig)
            # x_max_value = int(max(pop_gt_test_fig))
            # x_min_value = min(pop_gt_test_fig)
            # y_max_value = int(max(pred_test_fig))
            # smaller_max_value = min(x_max_value, y_max_value)
            #
            # interp_func = interpolate.interp1d(pop_gt_test_fig, pred_test_fig)
            #
            # interp_func_x = np.array([x_min_value, x_max_value])
            # interp_func_y = interp_func(interp_func_x)
            # plt.plot([0, smaller_max_value], [0, smaller_max_value],
            #          color='black', linewidth=1, linestyle='-', label='ground_truth')
            # # plt.plot(interp_func_x, interp_func_y, color='orange', linewidth=1, linestyle='-', label='fitted curve ')
            # plt.legend()

        ################################ save result for all item ################################
        # if mse < mse_for_pred_final[0]:
        #     print('found first best mse!')
        #     mse_for_pred_final[2] = mse_for_pred_final[1]
        #     pred_final_3 = pred_final_2
        #     mse_for_pred_final[1] = mse_for_pred_final[0]
        #     pred_final_2 = pred_final_1
        #     mse_for_pred_final[0] = mse
        #     mae_final, mse_final, item_final, pred_final_1, pop_gt_final = evaluate.pred_for_all_item(model, config)
        # elif mse < mse_for_pred_final[1]:
        #     print('found second best mse!')
        #     mse_for_pred_final[2] = mse_for_pred_final[1]
        #     pred_final_3 = pred_final_2
        #     mse_for_pred_final[1] = mse
        #     mae_final, mse_final, item_final, pred_final_2, pop_gt_final = evaluate.pred_for_all_item(model, config)
        # elif mse < mse_for_pred_final[2]:
        #     print('found third best mse!')
        #     mse_for_pred_final[2] = mse
        #     mae_final, mse_final, item_final, pred_final_3, pop_gt_final = evaluate.pred_for_all_item(model, config)

# save loss fig
# plt.savefig(config.fig_name_pre + '_' + str(best_mse) + '-visual.jpg')
# plt.clf()
# plt.figure(figsize=(7, 7))
# plt.title('Train/test loss')
# plt.xlabel('epoch')
#
# plt.ylabel('loss')
# plt.plot(range(len(list_train_loss)), list_train_loss, 'orange', label='train_loss')
# plt.plot(range(len(list_test_loss)), list_test_loss, 'darkblue', label='test_loss')
# plt.legend()
# plt.savefig(config.fig_name_pre + '_' + str(best_mse) + '-loss.jpg')
# print('\nDone, best epoch: {}, best MAE: {:.5f}, best MSE: {:.5f}'.format(best_epoch, best_mae, best_mse))
#
# # save pop result
# if len(pred_final_1) > 0:
#     with open('{}_{}_{}_1.pop'.format(config.result_path, str(best_mae)[:8], str(best_mse)[:8]), 'w') as f_result:
#         # for i in range(len(item_test)):
#         #     f_result.write('{} {} {}\n'.format(item_test[i], best_pred_test[i], best_pop_gt_test[i]))
#         for i in range(len(item_final)):
#             f_result.write('{} {} {}\n'.format(item_final[i], pred_final_1[i], pop_gt_final[i]))
#     print('{}_{}_{}_1.pop'.format(config.result_path, str(best_mae)[:8], str(best_mse)[:8]).split('/')[-1])
#
# if len(pred_final_2) > 0:
#     with open('{}_{}_{}_2.pop'.format(config.result_path, str(best_mae)[:8], str(best_mse)[:8]), 'w') as f_result:
#         # for i in range(len(item_test)):
#         #     f_result.write('{} {} {}\n'.format(item_test[i], best_pred_test[i], best_pop_gt_test[i]))
#         for i in range(len(item_final)):
#             f_result.write('{} {} {}\n'.format(item_final[i], pred_final_2[i], pop_gt_final[i]))
#     print('{}_{}_{}_2.pop'.format(config.result_path, str(best_mae)[:8], str(best_mse)[:8]).split('/')[-1])
#
# if len(pred_final_3) > 0:
#     with open('{}_{}_{}_3.pop'.format(config.result_path, str(best_mae)[:8], str(best_mse)[:8]), 'w') as f_result:
#         # for i in range(len(item_test)):
#         #     f_result.write('{} {} {}\n'.format(item_test[i], best_pred_test[i], best_pop_gt_test[i]))
#         for i in range(len(item_final)):
#             f_result.write('{} {} {}\n'.format(item_final[i], pred_final_3[i], pop_gt_final[i]))
#     print('{}_{}_{}_3.pop'.format(config.result_path, str(best_mae)[:8], str(best_mse)[:8]).split('/')[-1])

print('\nbest_epoch: {}'.format(best_epoch))
print('HR@5: {}, HR@10: {}, NDCG@5: {}, NDCG@10: {}, test_mse@{}'
      .format(best_hr[0], best_hr[1], best_ndcg[0], best_ndcg[1], best_test_mse))
with open('{}.pop'.format(config.result_path), 'w') as f_result:
    for i in range(len(best_pred)):
        f_result.write('{} {}\n'.format(i, best_pred[i]))
print('{}.pop'.format(config.result_path).split('/')[-1])
