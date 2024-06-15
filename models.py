import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


class OverAll(nn.Module):
    def __init__(self, node_size, node_hidden,
                 rel_size,
                 time_size,
                 triple_size,
                 rel_matrix,
                 ent_matrix,
                 time_matrix,
                 batch_size,
                 dropout_rate=0, depth=2,
                 gamma = 3,
                 device='cpu',
                 ):
        super(OverAll, self).__init__()
        self.dropout_rate = dropout_rate
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size
        self.alpha = 0.7

        self.e_encoder = GraphAttention(node_size, rel_size, triple_size, time_size, depth=depth, device=device,
                                        dim=node_hidden)
        self.t_encoder = GraphAttention(node_size, rel_size, triple_size, time_size, depth=depth, device=device,
                                dim=node_hidden)


        self.ent_adj = self.get_spares_matrix_by_index(ent_matrix, (node_size, node_size)) 
        self.rel_adj = self.get_spares_matrix_by_index(rel_matrix, (node_size, rel_size))
        self.time_adj = self.get_spares_matrix_by_index(time_matrix, (node_size, time_size))
        self.ent_emb_r = self.init_emb(node_size, node_hidden)  
        self.ent_emb_t = self.init_emb(node_size, node_hidden) 
        self.time_emb = self.init_emb(time_size, node_hidden)
        self.rel_emb = self.init_emb(rel_size, node_hidden)

        self.ent_adj, self.rel_adj, self.time_adj = map(lambda x: x.to(device), [self.ent_adj, self.rel_adj, self.time_adj])
    
    def forward(self, inputs):
        ent_feature_r = torch.matmul(self.ent_adj, self.ent_emb_r) 
        rel_feature = torch.matmul(self.rel_adj, self.rel_emb) 
        ent_feature_t = torch.matmul(self.ent_adj, self.ent_emb_t)
        time_feature = torch.matmul(self.time_adj, self.time_emb)

        adj_input = inputs[0]
        r_index = inputs[1] 
        r_val = inputs[2] 
        t_index = inputs[3]

        opt_r = [self.rel_emb, adj_input, r_index, r_val]
        opt_t = [self.time_emb, adj_input, t_index, r_val]

        ent_r_features = self.e_encoder([ent_feature_r] + opt_r)
        r_features = self.e_encoder([rel_feature] + opt_r)
        ent_t_features = self.t_encoder([ent_feature_t] + opt_t,1)
        t_features = self.t_encoder([time_feature] + opt_t,1)
        output_e_r = torch.cat((ent_r_features,r_features), dim=-1)
        output_e_t = torch.cat((ent_t_features,t_features), dim=-1)
        output_e_r = F.dropout(output_e_r, p=self.dropout_rate, training=self.training)
        output_e_t = F.dropout(output_e_t, p=self.dropout_rate, training=self.training)
        return output_e_r,output_e_t



    @staticmethod
    def get_spares_matrix_by_index(index, size):
        index = torch.LongTensor(index)
        adj = torch.sparse.FloatTensor(torch.transpose(index, 0, 1),
                                       torch.ones_like(index[:, 0], dtype=torch.float), size)
        return torch.sparse.softmax(adj, dim=1) 
    @staticmethod
    def init_emb(*size):
        entities_emb = nn.Parameter(torch.randn(size))
        torch.nn.init.xavier_normal_(entities_emb)
        return entities_emb

    def align_loss(self, align_input, embedding):
        def _cosine(x):
            dot1 = torch.bmm(x[0].unsqueeze(1), x[1].unsqueeze(2)).squeeze()
            dot2 = torch.bmm(x[0].unsqueeze(1), x[0].unsqueeze(2)).squeeze()
            dot3 = torch.bmm(x[1].unsqueeze(1), x[1].unsqueeze(2)).squeeze()
            max_ = torch.max(torch.sqrt(dot2 * dot3), torch.finfo(dot2.dtype).eps)
            return dot1 / max_
        def l1(ll, rr):
            return torch.sum(torch.abs(ll - rr), dim=-1, keepdim=True)
        def l2(ll, rr):
            return torch.sum(torch.square(ll - rr), dim=-1, keepdim=True)

        l, r, fl, fr = embedding[torch.tensor(align_input[:, 0])], embedding[torch.tensor(align_input[:, 1])], embedding[torch.tensor(align_input[:, 2])], embedding[torch.tensor(align_input[:, 3])]
        loss = F.relu(self.gamma + l1(l, r) - l1(l, fr)) + F.relu(self.gamma + l1(l, r) - l1(fl, r))
        loss = torch.sum(loss, dim=0, keepdim=True) / self.batch_size
        return loss
    



class GraphAttention(nn.Module):
    def __init__(self, node_size, rel_size, triple_size, time_size,
                 activation=torch.relu,
                 attn_heads=1, dim=100,
                 depth=1, device='cpu'):
        super(GraphAttention, self).__init__()
        self.node_size = node_size
        self.rel_size = rel_size
        self.time_size = time_size
        self.triple_size = triple_size
        self.activation = activation
        self.attn_heads = attn_heads
        self.depth = depth
        self.device = device
        node_F = dim
       
        self.attn_kernels = nn.ParameterList([OverAll.init_emb(3*node_F ,1) for i in range(self.depth*self.attn_heads)])
        self.all_attn_kernels = []
        for i in range(0, self.depth*self.attn_heads, self.attn_heads):
            temp = []
            for j in range(self.attn_heads): 
                temp.append(self.attn_kernels[i+j])
            self.all_attn_kernels.append(temp)

    def forward(self, inputs, rel_or_time=0):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1] 
        adj_index = inputs[2]  
        index = torch.tensor(adj_index, dtype=torch.int64)
        index = index.to(self.device)

        sparse_indices = inputs[3]  
        sparse_val = inputs[4] 

        features = self.activation(features)
        outputs.append(features)

        for l in range(self.depth):
            features_list = []
            for head in range(self.attn_heads):
                attention_kernel = self.all_attn_kernels[l][head]
                col = self.rel_size if rel_or_time == 0 else self.time_size
                rels_sum = torch.sparse.FloatTensor(
                    torch.transpose(torch.LongTensor(sparse_indices), 0, 1), 
                    torch.FloatTensor(sparse_val),
                    (self.triple_size, col))  

                rels_sum = rels_sum.to(self.device)
                rels_sum = torch.matmul(rels_sum, rel_emb) 

                neighs = features[index[:, 1]]
                selfs = features[index[:, 0]]

                rels_sum = F.normalize(rels_sum, p=1, dim=1)
                neighs = neighs - 2 * torch.sum(neighs * rels_sum, 1, keepdim=True) * rels_sum

                att1 = torch.squeeze(torch.matmul(torch.cat([selfs,neighs,rels_sum],dim=1), attention_kernel), dim=-1)
                att = torch.sparse.FloatTensor(torch.transpose(index, 0, 1), att1, (self.node_size, self.node_size))
                att = torch.sparse.softmax(att, dim=1)
                new_features = torch_scatter.scatter_add(
                    torch.transpose(neighs * torch.unsqueeze(att.coalesce().values(), dim=-1), 0, 1),
                    index[:, 0])
                
                new_features = torch.transpose(new_features, 0, 1)
                features_list.append(new_features)

            features = torch.cat(features_list)
            features = self.activation(features)
            outputs.append(features)
        outputs = torch.cat(outputs, dim=1)

        return outputs


