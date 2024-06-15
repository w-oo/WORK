import numpy as np
import torch
import torch.nn.functional as F
from utils import *
from CSLS import *
from models import OverAll
from args import args
from dto import *
from sinkhorn import *
from loss import *

dataset = args.data   
device = torch.device(args.gpu)
if dataset == 'ICEWS05-15/':
    print("icew_simt")
    file_name = "./data/"+dataset+"simt/simt_"+ dataset[0:-1] +"_" + str(args.seed) + ".npy"
    epoch = 600
    test_epoch = 200
else:
    print("yw_simt")
    file_name = "./data/"+dataset+"simt/simt_"+ dataset[0:-1] +"_" + str(args.seed) + ".npy"
    epoch = 1200
    test_epoch = 300

quadruples,train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features,time_dict,t_index,time_features = load_data("./data/"+dataset,ratio=args.seed)



pair_mt = get_simt(file_name,time_dict,dev_pair)
adj_matrix = np.stack(adj_matrix.nonzero(),axis = 1) 
rel_matrix, _ = np.stack(rel_features.nonzero(),axis = 1),rel_features.data 
ent_matrix, _ = np.stack(adj_features.nonzero(),axis = 1),adj_features.data 
time_matrix, _ = np.stack(time_features.nonzero(),axis = 1),time_features.data 


node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
time_size = time_features.shape[1]
triple_size = len(adj_matrix) 
batch_size = node_size

def get_embedding():
    inputs = [adj_matrix, r_index, r_val, t_index, rel_matrix, ent_matrix]
    ent_rel_emb,ent_time_emb = model(inputs)
    return ent_rel_emb,ent_time_emb



def SinkHorn_test(t_sims=pair_mt):
    model.eval()
    with torch.no_grad():

        ent_rel,ent_time = get_embedding()
        ent_time = ent_time.cpu().numpy()
        ent_rel = ent_rel.cpu().numpy()

        Lvec = np.array([ent_rel[e1] for e1, e2 in dev_pair])
        Rvec = np.array([ent_rel[e2] for e1, e2 in dev_pair])
        Lvec = Lvec / np.linalg.norm(Lvec,axis=-1,keepdims=True)
        Rvec = Rvec / np.linalg.norm(Rvec,axis=-1,keepdims=True)

        LTvec = np.array([ent_time[e1] for e1, e2 in dev_pair])
        RTvec = np.array([ent_time[e2] for e1, e2 in dev_pair])
        LTvec = LTvec / np.linalg.norm(LTvec,axis=-1,keepdims=True)
        RTvec = RTvec / np.linalg.norm(RTvec,axis=-1,keepdims=True)

        sim_mat = get_sinkhorn_mat(Lvec,Rvec,LTvec,RTvec,t_sims)  
        _,hits1 = eval_alignment_by_sinkhorn_sim_mat(sim_mat,[1, 5, 10])
    model.train()
    return None

def CSLS_test(thread_number = 16, csls=10,accurate = True,m=pair_mt):
    model.eval()
    with torch.no_grad():
        vec = get_embedding().cpu().numpy()
        Lvec = np.array([vec[e1] for e1, e2 in dev_pair])
        Rvec = np.array([vec[e2] for e1, e2 in dev_pair])
        Lvec = Lvec / np.linalg.norm(Lvec,axis=-1,keepdims=True)
        Rvec = Rvec / np.linalg.norm(Rvec,axis=-1,keepdims=True)
        eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], thread_number, m, csls=csls, accurate=accurate)
    model.train()
    return None

def get_train_set(batch_size = batch_size):
    negative_ratio =  batch_size // len(train_pair) + 1
    train_set = np.reshape(np.repeat(np.expand_dims(train_pair,axis=0),axis=0,repeats=negative_ratio),newshape=(-1,2))
    np.random.shuffle(train_set); train_set = train_set[:batch_size]
    train_set = np.concatenate([train_set,np.random.randint(0,node_size,train_set.shape)],axis = -1)
    return train_set





rest_set_1 = [e1 for e1, e2 in dev_pair]
rest_set_2 = [e2 for e1, e2 in dev_pair]
np.random.shuffle(rest_set_1) 
np.random.shuffle(rest_set_2) 


print(epoch)
model = OverAll(
            node_size=node_size,
            rel_size=rel_size,
            time_size=time_size,
            node_hidden=args.dim,
            rel_matrix=rel_matrix, ent_matrix=ent_matrix,time_matrix=time_matrix,
            triple_size=triple_size, 
            depth=args.depth,batch_size=batch_size,
            dropout_rate=args.dropout,gamma=args.gamma,
            device=device)
model = model.to(device)
align_multi_loss_layer = CustomMultiLossLayer(loss_num=2).to(device)
params = [{"params":
                list(model.parameters()) +
                list(align_multi_loss_layer.parameters())
            }]
optimizer = torch.optim.RMSprop(params, lr=args.lr, weight_decay=0)
for name, param in model.named_parameters():
    print(name, param.shape)


for turn in range(5):
    print("iteration %d start."%turn)
    max_index = -1
    for i in range(epoch):
        train_set1 = get_train_set()
        train_set2 = get_train_set()
        inputs = [adj_matrix, r_index, r_val, t_index, rel_matrix, ent_matrix]
        ent_r_emb,ent_t_emb = model(inputs)
        loss_ent_r = model.align_loss(train_set1, ent_r_emb)
        loss_ent_t = model.align_loss(train_set2, ent_t_emb)
        loss = align_multi_loss_layer([loss_ent_r, loss_ent_t])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i%test_epoch == test_epoch-1:
            max_index = SinkHorn_test()

    model.eval()
    with torch.no_grad():
        new_pair = []
        ent_rel,ent_time = get_embedding()
        ent_rel = ent_rel.cpu().numpy()
        ent_time = ent_time.cpu().numpy()

        Lvec = np.array([ent_rel[e] for e in rest_set_1])
        Rvec = np.array([ent_rel[e] for e in rest_set_2])
        Lvec = Lvec / np.linalg.norm(Lvec,axis=-1,keepdims=True)
        Rvec = Rvec / np.linalg.norm(Rvec,axis=-1,keepdims=True)

        LTvec = np.array([ent_time[e] for e in rest_set_1])
        RTvec = np.array([ent_time[e] for e in rest_set_2])
        LTvec = LTvec / np.linalg.norm(LTvec,axis=-1,keepdims=True)
        RTvec = RTvec / np.linalg.norm(RTvec,axis=-1,keepdims=True)
        

        t1 = [list2dict(time_dict[e1]) for e1 in rest_set_1]
        t2 = [list2dict(time_dict[e2]) for e2 in rest_set_2]

        pair = []

        for i in range(len(rest_set_1)):
            pair.append([rest_set_1[i],rest_set_2[i]])
        m1 = thread_sim_matrix(t1,t2)

        sim_mat_1 = get_sinkhorn_mat(Lvec,Rvec,LTvec,RTvec,m1)
  
        del m1
        A,_ = eval_alignment_by_sinkhorn_sim_mat(sim_mat_1,[1, 5, 10],True, False)

        del sim_mat_1
        m2 = thread_sim_matrix(t2,t1)

        sim_mat_2 = get_sinkhorn_mat(Rvec,Lvec,RTvec,LTvec,m2)

        del m2
        B,_ = eval_alignment_by_sinkhorn_sim_mat(sim_mat_2,[1, 5, 10],True, False)

        del sim_mat_2

        A = sorted(list(A)); B = sorted(list(B))
        for a,b in A:
            if  B[b][1] == a:
                new_pair.append([rest_set_1[a],rest_set_2[b]])
        print("generate new semi-pairs: %d." % len(new_pair))
        
        train_pair = np.concatenate([train_pair,np.array(new_pair)],axis = 0)
        for e1,e2 in new_pair:
            if e1 in rest_set_1:
                rest_set_1.remove(e1) 
            
        for e1,e2 in new_pair:
            if e2 in rest_set_2:
                rest_set_2.remove(e2) 
    model.train()



