import torch
import torch.nn as nn
import torch.nn.functional as F

class ExplainDrug(nn.Module):
    """
    The ExplainDrug class
    """

    def __init__(self, num_diag, num_pro, num_asp=1958, e_dim=64):
        super(ExplainDrug, self).__init__()
        self.num_asp = num_asp  # number of aspects
        self.asp_emb = Aspect_emb(num_asp, 2 * e_dim)
        self.mlp = nn.Sequential(nn.Linear(2 * e_dim, 3*e_dim), nn.ReLU(), nn.Linear(3*e_dim, num_asp))
        self.e_dim = e_dim
        self.mapping = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 131))

    def forward(self, x, y, asp):

        diag_latent = x
        pro_latent = y
        patient_representations = torch.cat([diag_latent, pro_latent], dim=-1)
        if patient_representations.dim() == 1:
            patient_representations = patient_representations.unsqueeze(0)
        patient_representations = patient_representations[-1 : ] # 将patient_representations中的最后一个元素（或者称为最后一个值）提取出来，作为本次患者表示。
        # out = self.mapping(query)
        # TODO:detach
        detached_query_latent = patient_representations.detach()  # gradient shielding 可解释模块中患者嵌入不进行反向传播

        asp_latent = self.asp_emb(asp.unsqueeze(0))

        factor = F.softmax(self.mlp(detached_query_latent), dim=-1).unsqueeze(-1)

        patient_asp = torch.bmm(asp_latent.permute(0, 2, 1), factor).squeeze(-1)

        # # cosine similarity between patient_asp and patient
        sim = - F.cosine_similarity(patient_asp, detached_query_latent, dim=-1)
        return sim

class Aspect_emb(nn.Module):
    """
    module to embed each aspect to the latent space.
    """

    def __init__(self, num_asp, e_dim):
        super(Aspect_emb, self).__init__()
        # 使用参数化的权重矩阵W来表示所有方面的潜在表示
        self.num_asp = num_asp
        self.W = nn.Parameter(torch.randn(num_asp, e_dim))  # 包含所有方面的潜在表示

    def forward(self, x):
        shape = x.shape
        x = x.reshape([x.shape[0], x.shape[1], 1])
        # 将x在最后一个维度进行扩展以匹配权重矩阵W的维度
        x = x.expand(-1, -1, self.W.shape[1]) # 将x与self.W的形状对齐，以便后续的矩阵乘法。
        # 逐元素相乘，得到每个方面的潜在表示
        asp_latent = torch.mul(x, self.W)  # [1, num_asp, e_dim]
        # 可选择对每个方面的潜在表示进行标准化
        # asp_latent = F.normalize(asp_latent, p=2, dim=2)

        return asp_latent

class ExpDrug(nn.Module):
    def __init__(
        self,
        vocab_size,
        ehr_adj,
        ddi_adj,
        ddi_mask_H,
        emb_dim=256,
        device=torch.device("cpu:0"),
    ):
        super(ExpDrug, self).__init__()

        self.device = device
        self.emb_dim = emb_dim

        # pre-embedding

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(3)]
        )
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(3)]
        )
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim))


        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        self.init_weights()

        # aspect
        self.exp_diag = ExplainDrug(num_diag=4233, num_pro=1958, num_asp=1958, e_dim=64)
        self.exp_pro = ExplainDrug(num_diag=4233, num_pro=1430, num_asp=1430, e_dim=64)
        self.exp_med = ExplainDrug(num_diag=4233, num_pro=131, num_asp=131, e_dim=64)
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(1, emb_dim))
        self.mapping = nn.Sequential(nn.Linear(3*emb_dim, 64), nn.ReLU(), nn.Linear(64, 131))
        self.map_vocab_size = nn.Sequential(
            nn.Linear(2*emb_dim, vocab_size[2])
        )

    def forward(self, input, diag, pro, med, step):

        # patient health representation
        diag_seq = []
        proc_seq = []
        med_seq = []

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:


            diag_1 = sum_embedding(
                self.dropout(
                    self.embeddings[0](
                        torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )  # (1,1,dim)
            proc_1 = sum_embedding(
                self.dropout(
                    self.embeddings[1](
                        torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )



            diag_seq.append(diag_1)
            proc_seq.append(proc_1)

        for idx, adm in enumerate(input):
            if len(input) <= 1 or idx==0:
                med_1 = torch.zeros((1, 1, self.emb_dim)).to(self.device)
            else:
                 adm[2] = input[idx - 1][2][:]
                 a = adm[2]
                 med_1 = sum_embedding(
                     self.dropout(
                         self.embeddings[2](torch.LongTensor(adm[2]).unsqueeze(dim=0).to(self.device))))
            med_seq.append(med_1)




        diag_seq = torch.cat(diag_seq, dim=1)  # (1,seq,dim)
        proc_seq = torch.cat(proc_seq, dim=1)  # (1,seq,dim)
        med_seq = torch.cat(med_seq, dim=1)

        o1, h1 = self.encoders[0](diag_seq)
        o2, h2 = self.encoders[1](proc_seq)
        o3, h3= self.encoders[2](med_seq)

        o1 = o1.squeeze(dim=0).squeeze(dim=0)
        o2 = o2.squeeze(dim=0).squeeze(dim=0)
        o3 = o3.squeeze(dim=0).squeeze(dim=0)

        diag = diag.to(self.device)
        pro = pro.to(self.device)
        med = med.to(self.device)
        diag_sim = self.exp_diag(o1, o2, diag[step])
        pro_sim = self.exp_pro(o1, o2, pro[step])
        med_sim = self.exp_med(o1, o2, med[step])

        sim = (diag_sim + pro_sim + med_sim)/3.0

        query = torch.cat([o1, o2, o3], dim=-1).squeeze(0)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query = query[-1:]
        out = self.mapping(query)

        neg_pred_prob = F.sigmoid(out)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = F.sigmoid(neg_pred_prob.mul(self.tensor_ddi_adj)).sum()
        return out, batch_neg, sim

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)


