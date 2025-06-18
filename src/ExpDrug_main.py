import csv
import math

import dill
import numpy as np
import argparse
from collections import defaultdict
from sklearn.metrics import jaccard_score
from torch.optim import Adam
import os
import torch
import time
from ExpDrug_models import ExpDrug
from seed import set_seed
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, buildMPNN
import torch.nn.functional as F

# torch.manual_seed(1203)
# np.random.seed(2048)
# torch.set_num_threads(30)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# setting
model_name = "ExplainDrug"
resume_path = "Epoch_41_TARGET_0.06_JA_0.5871_DDI_0.0444.model"
if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--Test", action="store_true", default=False, help="test mode")
parser.add_argument("--model_name", type=str, default=model_name, help="model name")
parser.add_argument("--resume_path", type=str, default=resume_path, help="resume path")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--target_ddi", type=float, default=0.065, help="target ddi")
parser.add_argument("--kp", type=float, default=0.05, help="coefficient of P signal")
parser.add_argument("--dim", type=int, default=64, help="dimension")
parser.add_argument("--cuda", type=int, default=0, help="which cuda")
parser.add_argument("--a", type=float, default=0.9, help="coefficient of a")

args = parser.parse_args()

# evaluate
def eval(model, data_eval, voc_size, epoch, diag_eval, pro_eval, med_eval):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input):
            target_output, _, _= model(input[: adm_idx + 1], diag_eval, pro_eval, med_eval, step)

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path="../data/data_new/ddi_A_final.pkl")

    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )


def main():

    # load data
    data_path = "../data/data_new/records_final_131.pkl"
    # TODO: Shuffule-data131
    # data_path = "../random_bigdata131_5/data/records_final_131_shuffle5.pkl"
    voc_path = "../data/data_new/voc_final.pkl"


    ddi_adj_path = "../data/data_new/ddi_A_final.pkl"
    ddi_mask_path = "../data/data_new/ddi_mask_H.pkl"
    molecule_path = "../data/data_new/idx2drug.pkl"



    diag_path = "../data/data_new/diag_new.pkl"  # 新增diag
    pro_path = "../data/data_new/pro_new.pkl"  # 新增pro
    med_path = "../data/data_new/med131_new.pkl"  # 新增med

    ddi_A_final_path = "../data/data_new/ddi_A_final.pkl"
    # TODO :ehr
    ehr_adj_path = '../data/data_new/ehr_adj_final.pkl'



    device = torch.device("cuda:{}".format(args.cuda))

    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    data = dill.load(open(data_path, "rb"))
    molecule = dill.load(open(molecule_path, "rb"))

    diag_new = dill.load(open(diag_path, "rb")) # 新增diag
    pro_new = dill.load(open(pro_path, "rb")) # 新增peo
    med_new= dill.load(open(med_path, "rb")) # 新增med



    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]

    # data-split
    split_point = int(len(diag_new) * 2 / 3)
    diag = diag_new[:split_point]
    eval_len = int(len(diag_new[split_point:]) / 2)
    diag_test = diag_new[split_point: split_point + eval_len]
    diag_eval = diag_new[split_point + eval_len:]

    split_point = int(len(pro_new) * 2 / 3)
    pro = pro_new[:split_point]
    eval_len = int(len(diag_new[split_point:]) / 2)
    pro_test = pro_new[split_point: split_point + eval_len]
    pro_eval = pro_new[split_point + eval_len:]

    split_point = int(len(med_new) * 2 / 3)
    med = med_new[:split_point]
    eval_len = int(len(med_new[split_point:]) / 2)
    med_test = med_new[split_point: split_point + eval_len]
    med_eval = med_new[split_point + eval_len:]


    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = ExpDrug(
        voc_size,
        # TODO :ehr
        ehr_adj,
        ddi_adj,
        ddi_mask_H,

        emb_dim=args.dim,
        device=device,
    )
    # model.load_state_dict(torch.load(open(args.resume_path, 'rb')))

    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, "rb")))
        model.to(device=device)
        tic = time.time()

        result = []
        for _ in range(10):
            test_sample = np.random.choice(
                data_test, round(len(data_test) * 1), replace=True
            )
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
                model, data_test, voc_size, 0,diag_test,pro_test,med_test
            )
            result.append([ddi_rate, ja, avg_f1, prauc, avg_med])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print(outstring)

        print("test time: {}".format(time.time() - tic))
        return

    model.to(device=device)
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    # start iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0


    EPOCH = 50
    for epoch in range(EPOCH):
        tic = time.time()
        print("\nepoch {} --------------------------".format(epoch + 1))

        model.train()
        for step, input in enumerate(data_train):

            loss = 0
            for idx, adm in enumerate(input):

                seq_input = input[: idx + 1]
                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for idx, item in enumerate(adm[2]):
                    loss_multi_target[0][idx] = item

                result, loss_ddi, sim = model(seq_input, diag, pro, med, step)      # 新增diag, pro, med,返回sim

                loss_bce = F.binary_cross_entropy_with_logits(
                    result, torch.FloatTensor(loss_bce_target).to(device))

                loss_multi = F.multilabel_margin_loss(
                    F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device)
                )

                result = F.sigmoid(result).detach().cpu().numpy()[0]
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0]
                current_ddi_rate = ddi_rate_score(
                    [[y_label]], path=ddi_A_final_path
                )

                if current_ddi_rate <= args.target_ddi:
                    loss = (0.95 * loss_bce + 0.05 * loss_multi) * args.a + sim * (1 - args.a)

                else:
                    beta = max(0, 1 - np.tanh(current_ddi_rate))
                    loss = (beta * (0.95 * loss_bce + 0.05 * loss_multi)+ (1 - beta) * loss_ddi ) * args.a + sim * (1 - args.a)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint("\rtraining step: {} / {}".format(step, len(data_train)))

        print()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
            model, data_eval, voc_size, epoch, diag_eval, pro_eval, med_eval
        )
        print(
            "training time: {}, test time: {}".format(
                time.time() - tic, time.time() - tic2
            )
        )

        history["ja"].append(ja)
        history["ddi_rate"].append(ddi_rate)
        history["avg_p"].append(avg_p)
        history["avg_r"].append(avg_r)
        history["avg_f1"].append(avg_f1)
        history["prauc"].append(prauc)
        history["med"].append(avg_med)

        if epoch >= 5:
            print(
                "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                    np.mean(history["ddi_rate"][-5:]),
                    np.mean(history["med"][-5:]),
                    np.mean(history["ja"][-5:]),
                    np.mean(history["avg_f1"][-5:]),
                    np.mean(history["prauc"][-5:]),
                )
            )

        torch.save(
            model.state_dict(),
            open(
                os.path.join(
                    "saved",
                    args.model_name,
                    "Epoch_{}_TARGET_{:.2}_JA_{:.4}_DDI_{:.4}.model".format(
                        epoch, args.target_ddi, ja, ddi_rate
                    ),
                ),
                "wb",
            ),
        )

        # if epoch != 0 and best_ja < ja:
        #     best_epoch = epoch+1
        #     best_ja = ja
        if best_ja < ja:
            best_epoch = epoch+1
            best_ddi_rate = round(ddi_rate, 4)
            best_ja = round(ja, 4)
            best_prauc = round(prauc, 4)
            best_avg_p = round(avg_p, 4)
            best_avg_r = round(avg_r, 4)
            best_avg_f1 = round(avg_f1, 4)
            best_avg_med= round(avg_med, 4)

        # print("best_epoch: {}".format(best_epoch))
        print(
            "best_epoch: {}, best_ddi_rate: {}, best_ja: {}, best_avg_f1: {}, best_prauc: {}, best_avg_med: {}, best_avg_p: {}, best_avg_r: {}"
                .format(best_epoch, best_ddi_rate, best_ja, best_avg_f1, best_prauc, best_avg_med, best_avg_p,
                        best_avg_r))

    dill.dump(
        history,
        open(
            os.path.join(
                "saved", args.model_name, "history_{}.pkl".format(args.model_name)
            ),
            "wb",
        ),
    )

    save_dir = 'saved/best_epoch.csv'
    # 检查文件是否存在
    if not os.path.exists(save_dir) or os.path.getsize(save_dir) == 0:
        with open(save_dir, 'a+', newline='') as logfile:
            logwriter = csv.DictWriter(logfile,
                                       fieldnames=['lr', 'a',  'best_epoch', 'best_ddi_rate', 'best_ja', 'best_avg_f1',
                                                   'best_prauc', 'best_avg_med', 'best_avg_p', 'best_avg_r'])
            logwriter.writeheader()

    # 写入数据行
    with open(save_dir, 'a+', newline='') as logfile:
        logwriter = csv.DictWriter(logfile,
                                   fieldnames=['lr', 'a',  'best_epoch', 'best_ddi_rate', 'best_ja', 'best_avg_f1',
                                               'best_prauc', 'best_avg_med', 'best_avg_p', 'best_avg_r'])
        logdict = dict(lr=args.lr, a=args.a, best_epoch=best_epoch, best_ddi_rate=best_ddi_rate, best_ja=best_ja,
                       best_avg_f1=best_avg_f1, best_prauc=best_prauc, best_avg_med=best_avg_med, best_avg_p=best_avg_p,
                       best_avg_r=best_avg_r)
        logwriter.writerow(logdict)


if __name__ == "__main__":
    set_seed()
    main()
