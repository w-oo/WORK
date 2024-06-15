import numpy as np
import scipy.sparse as sp
import pickle
import os
from multiprocessing import Pool
from args import args
thread_num=20 

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

def load_alignment_pair(file_name):
    alignment_pair = []
    c = 0
    for line in open(file_name,'r'):
        e1,e2 = line.split()
        alignment_pair.append((int(e1),int(e2)))
    return alignment_pair

def get_matrix(triples,entity,rel,time): 
        ent_size = max(entity)+1
        rel_size = (max(rel) + 1)
        time_size = max(time) + 1
        print(ent_size,rel_size,time_size)
        adj_matrix = sp.lil_matrix((ent_size,ent_size)) 
        adj_features = sp.lil_matrix((ent_size,ent_size))
        radj = []
        rel_in = np.zeros((ent_size,rel_size)) 
        rel_out = np.zeros((ent_size,rel_size)) 
        time_dict = {}
        
        for i in range(max(entity)+1):
            adj_features[i,i] = 1 
            time_dict[i] = [] 
        time_link = np.zeros((ent_size,time_size))
        for h,r,t,tb,te in triples:        
            adj_matrix[h,t] = 1; adj_matrix[t,h] = 1  
            adj_features[h,t] = 1; adj_features[t,h] = 1 
            radj.append([h,t,r,tb]); radj.append([t,h,r+rel_size,te]);  
            rel_out[h][r] += 1; rel_in[t][r] += 1 
            time_link[h][te] +=1 ; time_link[h][tb] +=1
            time_link[t][tb] +=1 ; time_link[t][te] +=1 
            if (te==tb):
                time_dict[h].append(tb); time_dict[t].append(tb)  
            else:
                time_dict[h].append(tb); time_dict[t].append(tb)
                time_dict[h].append(te); time_dict[t].append(te)

        count = -1
        s = set()
        d = {}
        r_index,t_index,r_val = [],[],[]
        for h,t,r,tau in sorted(radj,key=lambda x: x[0]*10e10+x[1]*10e5):
            if ' '.join([str(h),str(t)]) in s: 
                r_index.append([count,r])
                t_index.append([count,tau])
                r_val.append(1) 
                d[count] += 1 
            else:
                count += 1 
                d[count] = 1 
                s.add(' '.join([str(h),str(t)])) 
                r_index.append([count,r]) 
                t_index.append([count,tau])
                r_val.append(1) 
        for i in range(len(r_index)): 
            r_val[i] /= d[r_index[i][0]] 

        time_features  = time_link
        time_features = normalize_adj(sp.lil_matrix(time_features))         
        rel_features = np.concatenate([rel_in,rel_out],axis=1) 
        adj_features = normalize_adj(adj_features) 
        rel_features = normalize_adj(sp.lil_matrix(rel_features))  
        return adj_matrix,r_index,r_val,adj_features,rel_features,time_dict,time_features,t_index  
    
def load_quadruples(file_name):
    quadruples = []   
    entity = set() 
    rel = set([0]) 
    time = set()
    for line in open(file_name,'r'):
        items = line.split()
        if len(items) == 4:
            head,r,tail,t = [int(item) for item in items] 
            entity.add(head); entity.add(tail); rel.add(r); time.add(t)  
            quadruples.append((head,r,tail,t,t)) 
        else:
            head,r,tail,tb,te = [int(item) for item in items] 
            entity.add(head); entity.add(tail); rel.add(r); time.add(tb); time.add(te)  
            quadruples.append((head,r,tail,tb,te))  
    return entity,rel,time,quadruples 

def load_data(path, ratio=1000): 
    if args.data == 'ICEWS05-15/' and args.seed == 1000:
        print("icew1000")
        if os.path.exists(path+"graph_cache_yw1000.pkl"):
            return pickle.load(open(path+"graph_cache_yw1000.pkl","rb"))
    if args.data == 'ICEWS05-15/' and args.seed == 200:
        print("icew200")
        if os.path.exists(path+"graph_cache_icews200.pkl"):
            return pickle.load(open(path+"graph_cache_icews200.pkl","rb"))
    if args.data != 'ICEWS05-15/' and args.seed == 1000:
        print("yw1000")
        if os.path.exists(path+"graph_cache_yw1000.pkl"):
            return pickle.load(open(path+"graph_cache_yw1000.pkl","rb"))
    if args.data != 'ICEWS05-15/' and args.seed == 5000:
        print("yw5000")
        if os.path.exists(path+"graph_cache_yw5000.pkl"):
            return pickle.load(open(path+"graph_cache_yw5000.pkl","rb"))
    entity1,rel1,time1,quadruples1 = load_quadruples(path + 'triples_1')
    entity2,rel2,time2,quadruples2 = load_quadruples(path + 'triples_2')

    train_pair = load_alignment_pair(path + 'sup_pairs')
    dev_pair = load_alignment_pair(path + 'ref_pairs')
    dev_pair = train_pair[ratio:]+dev_pair 
    train_pair = train_pair[:ratio] 

    adj_matrix,r_index,r_val,adj_features,rel_features,time_dict,time_features,t_index = get_matrix(quadruples1+quadruples2,entity1.union(entity2),rel1.union(rel2),time1.union(time2)) 
    graph_data = [quadruples1 + quadruples2,np.array(train_pair),np.array(dev_pair),adj_matrix,np.array(r_index),np.array(r_val),adj_features,rel_features,time_dict,np.array(t_index),time_features]
    if args.data == 'ICEWS05-15/' and args.seed == 1000:
        pickle.dump(graph_data, open(path+"graph_cache_icews1000.pkl","wb"))
    if args.data == 'ICEWS05-15/' and args.seed == 200:
        pickle.dump(graph_data, open(path+"graph_cache_icews200.pkl","wb"))
    if args.data != 'ICEWS05-15/' and args.seed == 1000:
        pickle.dump(graph_data, open(path+"graph_cache_yw1000.pkl","wb"))
    if args.data != 'ICEWS05-15/' and args.seed == 5000:
        pickle.dump(graph_data, open(path+"graph_cache_yw5000.pkl","wb"))
    return graph_data
    
def get_hits(vec, test_pair, wrank = None, top_k=(1, 5, 10)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    
    Lvec = Lvec / np.linalg.norm(Lvec,axis=-1,keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec,axis=-1,keepdims=True)
    sim_o = -Lvec.dot(Rvec.T)
    sim = sim_o.argsort(-1)
    if wrank is not None:
        srank = np.zeros_like(sim)
        for i in range(srank.shape[0]):
            for j in range(srank.shape[1]):
                srank[i,sim[i,j]] = j
        rank = np.max(np.concatenate([np.expand_dims(srank,-1),np.expand_dims(wrank,-1)],-1),axis=-1)
        sim = rank.argsort(-1)
    top_lr = [0] * len(top_k)
    MRR_lr = 0
    for i in range(Lvec.shape[0]):
        rank = sim[i, :]
        rank_index = np.where(rank == i)[0][0]
        MRR_lr += 1/(rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    MRR_rl = 0
    sim = sim_o.argsort(0)
    for i in range(Rvec.shape[0]):
        rank = sim[:,i]
        rank_index = np.where(rank == i)[0][0]
        MRR_rl += 1/(rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('MRR: %.3f' % (MRR_lr / Lvec.shape[0]))  
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    print('MRR: %.3f' % (MRR_rl / Rvec.shape[0]))



def get_simt(file_name,time_dict,dev_pair):
    if os.path.exists(file_name): 
        pair_mt = np.load(file_name) 
    else: 
        pair_mt = pair_simt(time_dict,dev_pair) 
        np.save(file_name,pair_mt) 
    return pair_mt

def list2dict(time_list):
    
    dic={}
    for i in time_list:
        dic[i]=time_list.count(i)
    return dic 

def sim_matrix(t1,t2):
    
    
    size_t1 = len(t1)
    size_t2 = len(t2)
    matrix = np.zeros((size_t1,size_t2))
    for i in range(size_t1):
        len_a = sum(t1[i].values()) 
        for j in range(size_t2):
            len_b = sum(t2[j].values())
            len_ab = len_a + len_b 
            set_ab = {}
            set_ab = t1[i].keys() & t2[j].keys() 

            if (len(set_ab)==0): 
                matrix[i,j] = 0
                continue
            count = 0
            for k in set_ab: 
                count = count + (min(t1[i][k],t2[j][k])-1) 
            count = len(set_ab) + count 
            matrix[i,j] = (count*2) / len_ab 
    return matrix

def div_array(arr,n):
    
    arr_len = len(arr)
    k = arr_len // n
    ls_return = []
    for i in range(n-1):
        ls_return.append(arr[i*k:i*k+k])
    ls_return.append(arr[(n-1)*k:])
    return ls_return 

def thread_sim_matrix(t1,t2):
    pool = Pool(processes=thread_num)
    reses = list()
    tasks_t1 = div_array(t1,thread_num)    
    for task_t1 in tasks_t1: 
        reses.append(pool.apply_async(sim_matrix,args=(task_t1,t2)))
    pool.close()
    pool.join()
    matrix = None
    for res in reses:
        val = res.get() 
        if matrix is None:
            matrix = np.array(val) 
        else:
            matrix = np.concatenate((matrix,val),axis=0) 

    return matrix 

def pair_simt(time_dict,pair):
    
    t1 = [list2dict(time_dict[e1]) for e1, e2 in pair]
    t2 = [list2dict(time_dict[e2]) for e1, e2 in pair]
    m = thread_sim_matrix(t1,t2) 
    return m 









