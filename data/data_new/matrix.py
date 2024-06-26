import os
import dill
import torch

def main():
    data_path = "../data/data_new/records_final_131.pkl"
    data = dill.load(open(data_path, "rb"))
    rows = 6350
    d_cols = 1958   # d
    p_cols = 1430   # p
    m_cols = 131      # m
    diag = torch.zeros(rows, d_cols)   # d
    pro = torch.zeros(rows, p_cols)    # p
    med = torch.zeros(rows, m_cols)      # m
    for step, input in enumerate(data):


        for idx, adm in enumerate(input):
            d = adm[0]   # d
            p = adm[1]   # p
            m = adm[2]     # m
            for j in d:
                diag[step][j] = 1   # d

            for j in p:   # p
                pro[step][j] = 1    # p

            for j in m:     # m
                med[step][j] = 1      # m

    d_path = "../data/data_new/diag_new.pkl"   # d
    p_path = "../data/data_new/pro_new.pkl"    # p
    m_path = "../data/data_new/med131_new.pkl"      # m
    d_directory = os.path.dirname(d_path)
    p_directory = os.path.dirname(p_path)
    m_directory = os.path.dirname(m_path)
    # 创建目录（如果不存在）
    os.makedirs(d_directory, exist_ok=True)
    os.makedirs(p_directory, exist_ok=True)
    os.makedirs(m_directory, exist_ok=True)

    dill.dump(diag, open(d_path, "wb"))   # d
    dill.dump(pro, open(p_path, "wb"))    # p
    dill.dump(med, open(m_path, "wb"))      # m
    return


if __name__ == "__main__":
    main()