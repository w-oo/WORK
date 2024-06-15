import numpy as np
import time
import multiprocessing
from CSLS import div_list,cal_rank_by_sim_mat
import gc
from args import args


def get_sinkhorn_mat(Lvec,Rvec,LTvec,RTvec,t_sims):
    if args.CF_type == "TRO":
        sim_mat = np.exp((1-args.eta)*(1-args.omega)*np.matmul(Lvec,Rvec.T) + (1-args.eta)*args.omega*np.matmul(LTvec,RTvec.T) + args.eta*t_sims)*50
    else:
        sim_mat = np.exp((1-args.eta)*(1-args.omega)*np.matmul(Lvec,Rvec.T) + args.omega*np.matmul(LTvec,RTvec.T) + args.eta*t_sims)*50

    for k in range(10):
        sim_mat = sim_mat /  np.sum(sim_mat, axis=1, keepdims=True)
        sim_mat = sim_mat / np.sum(sim_mat, axis=0, keepdims=True)
    return sim_mat


def cal_sims(test_pair,feature):
    Lvec = np.array([feature[e1] for e1, e2 in test_pair])
    Rvec = np.array([feature[e2] for e1, e2 in test_pair])
    Lvec = Lvec / np.linalg.norm(Lvec,axis=-1,keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec,axis=-1,keepdims=True)
    return np.matmul(Lvec,Rvec.T)


def eval_alignment_by_sinkhorn_sim_mat(sims,top_k,accurate=True,output=True,nums_threads=16):
    t = time.time()
    ref_num = sims.shape[0]
    t_num = [0 for k in top_k]
    t_mean = 0
    t_mrr = 0
    t_prec_set = set()
    tasks = div_list(np.array(range(ref_num)), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_rank_by_sim_mat, (task, sims[task, :], top_k, accurate)))
    pool.close()
    pool.join()

    for res in reses:
        mean, mrr, num, prec_set = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += np.array(num)
        t_prec_set |= prec_set
    assert len(t_prec_set) == ref_num
    acc = np.array(t_num) / ref_num * 100
    for i in range(len(acc)):
        acc[i] = round(acc[i], 2)
    t_mean /= ref_num
    t_mrr /= ref_num
    if output:
        if accurate:
            print("accurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc, t_mean,t_mrr,time.time() - t))
        else:
            print("hits@{} = {}, time = {:.3f} s ".format(top_k, acc, time.time() - t))
    hits1 = acc[0]
    del sims
    gc.collect()
    return t_prec_set, hits1
  