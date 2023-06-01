"""
Created on Mon Mar 21 15:04:29 2022

@author: gmarchel
"""

import numpy as np
import torch
from scipy.special import logsumexp
import torch.nn as nn
import time
from scipy.special import factorial
import torch.nn.functional as F
import sys
from typing import List, Optional

# decomment if wanted : multiprocessing
from pathos.pools import ProcessPool as Pool

# decomment if wanted : multithreading
# from pathos.threading import ThreadPool as Pool


torch._C._jit_set_profiling_mode(False)

'''Building the neural networks:'''
'''Mult_alpha is a two layers fully connected Neural Net'''


class MultAlpha(torch.nn.Module):
    def __init__(self, input_size=3, hidden_size_1=500, hidden_size_2=250, output_size=3):
        super(MultAlpha, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        layer1_out = F.relu(self.linear1(x))
        layer2_out = F.relu(self.linear2(layer1_out))
        out = self.linear3(layer2_out)
        return out

    @staticmethod
    def compute_l1_loss(w):
        return torch.abs(w).sum()


''' three layers fully connected Neural Net'''


class MultTre(torch.nn.Module):
    def __init__(self, input_size=3, hidden_size_1=500, hidden_size_2=250, hidden_size_3=250, output_size=3):
        super(MultTre, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.linear4 = nn.Linear(hidden_size_3, output_size)

    def forward(self, x):
        layer1_out = F.relu(self.linear1(x))
        layer2_out = F.relu(self.linear2(layer1_out))
        layer3_out = F.relu(self.linear3(layer2_out))
        out = self.linear4(layer3_out)
        return out


''' custom class inherit from torch.nn.Module and make latent variables operations, also used in JIT scripting'''
''' latent_var0: initial value of the dynamic system (beta_0)'''
class Mupdate(torch.nn.Module):
    def __init__(self, network, latent_var: torch.Tensor, lr: float = 1e-5, gpu: Optional[int] = None,
                 pi: bool = False):
        super(Mupdate, self).__init__()
        self.softmax = nn.Softmax(0)
        self.lr = lr
        self.pi = pi
        tensor = torch.tensor(latent_var[:, :, 0])
        device = "cpu" if gpu is None else "cuda:" + str(gpu)
        self.network = network
        self.network.to(torch.device(device))
        self.latent_var = torch.tensor(latent_var, dtype=torch.float32, requires_grad=False, device=device)
        self.latent_var_0 = torch.tensor(
            (torch.log(torch.mean(tensor)), torch.log(1 - torch.mean(tensor))),
            requires_grad=True, dtype=torch.float32, device=device) if self.pi else torch.tensor(
            torch.log(torch.mean(tensor, dim=0)), requires_grad=True, dtype=torch.float32, device=device)
        self.U: int = self.latent_var.size(2)

    '''Computing the loss of beta: we want to minimize it'''
    @torch.jit.export
    def loss(self):
        prop_0 = self.softmax(self.latent_var_0)
        if self.pi:
            out = torch.sum(self.latent_var[:, :, 0] * torch.log(prop_0[0] + 1e-16)) + torch.sum(
                (1 - self.latent_var[:, :, 0]) * torch.log(
                    1 - prop_0[0] + 1e-16))
        else:
            out = torch.sum(torch.sum(self.latent_var[:, :, 0], 0) * torch.log(prop_0 + 1e-16))
        prev_ = self.latent_var_0
        for u in range(1, self.U):
            next_ = prev_ + self.network(prev_)
            propt_next_ = self.softmax(next_.reshape(-1, ))
            log_next_ = torch.log(propt_next_ + 1e-6) if self.pi else torch.log(propt_next_ + 1e-16)
            latent_var_u = self.latent_var[:, :, u]
            if self.pi:
                out += torch.sum(latent_var_u * log_next_[0]) + torch.sum((1 - latent_var_u) * torch.log(
                    1 - propt_next_[0] + 1e-16))
            else:
                out += torch.sum(torch.sum(latent_var_u, 0) * log_next_)
            prev_ = next_
        return -out

    ''' Function implementing the ODE system'''
    @torch.jit.export
    def get_next(self, axis_clusters: Optional[int] = None):
        if axis_clusters is None:
            out = torch.zeros(self.U, 2)
        else:
            out = torch.zeros(self.U, axis_clusters)
        out[0, :] = self.softmax(self.latent_var_0)  # softmax of the initial value
        prev_ = self.latent_var_0
        for u in range(1, self.U):
            next_ = prev_ + self.network(prev_)
            out[u, :] = self.softmax(next_.reshape(-1, ))
            prev_ = next_.clone()
        return out

'''Training the loop over the epochs'''
''' 
    opt: Adam optimizer '''
def mupdate(module, epochs: int = 100):
    opt = torch.optim.Adam(params=[module.latent_var_0] + list(module.network.parameters()), lr=module.lr)
    store_l = []  # To store the loss values
    for epoch in range(epochs):
        opt.zero_grad()  # Gradient computation
        l = module.loss()  # Loss computation
        store_l.append(l.item())
        l.backward()  # Backard propagation
        opt.step()  # Optimization step - > Loss minimization
        if epoch % 100 == 0:
            # Print the loss every 100 epochs
            print('epoch :', epoch, 'loss :', l.item())
            sys.stdout.flush()
    return store_l


def wrap_everything(conf: dict, jit=False):
    if 'i_inf_10' in conf and not (conf['i_inf_10']):
        return None, None
    else:
        start_time = time.time()
        if jit:
            module = torch.jit.load(conf['jit'] + '/scripted_mupdate_' + conf['name'] + '.pt')
        else:
            module = Mupdate(conf["network"], conf['latent_var'], conf['lr'], gpu=conf["gpu"], pi=conf['pi_ops'])
        store_l = mupdate(module, epochs=conf['epochs'])
        print("--- %s seconds ---" % (time.time() - start_time))
        est = module.get_next(conf['axis_cluster'])
        est = est.detach().numpy()
        return est, store_l


def do_jit_saving_op(config_model: dict):
    ''' save TorchScript versions of Mupdate classes (pi, alpha and beta)'''
    for key, value in config_model.copy().items():
        config_model[key].update({'scripted_network': torch.jit.script(value['network'])})
        config_model[key]['scripted_network'].save('./traced_func/nnModules/scripted_f_' + str(key) + '.pt')
        mupdate_y = Mupdate(config_model[key]['scripted_network'],
                            config_model[key]['latent_var'],
                            config_model[key]['lr'],
                            gpu=config_model[key]['gpu'],
                            pi=config_model[key]['pi_ops'])
        scripted_network = torch.jit.script(mupdate_y)
        scripted_network.save(config_model[key]["jit"] + '/scripted_mupdate_' + str(key) + '.pt')


''' Stream_DLBM is the main function:'''
'''
  X = our count data. MxPxU array cointaining the number of interaction between every row and column pair at any given time instants,
  Q = number of row clusters,
  L = number of column clusters,
  max_iter = number of iteration of the algorithm 
  alpha_init = UxQ matrix of initial values of alpha,
  beta_init = UxL matrix of initial values of beta,
  Lambda_init = QxL matrix of initial values of Lambda,
  pi = Ux2 matrix of initial values of pi. 
'''


# @profile
def stream_DLBM(X, Q, L, max_iter, alpha_init, beta_init, Lambda_init, pi):
    tests = {

        'multiprocessing': True,
        'do_jit': False, 
        'do_jit_saving': True,  # needed before doing "do_jit" for the first time
        'gpu': False  # gpu and jit are not working together in this script

    }

    M = X.shape[0]
    P = X.shape[1]
    U = X.shape[2]

    '''Data initialization '''
    e_it = 3
    alpha = np.copy(alpha_init)
    beta = np.copy(beta_init)
    Lambda = np.copy(Lambda_init)
    pi = np.copy(pi)

    # to avoid numerical problems, if the parameters are 0 then we transform it in 1e-16
    alpha[alpha < 1e-16] = 1e-16
    beta[beta < 1e-16] = 1e-16
    Lambda[Lambda < 1e-16] = 1e-16

    tau = np.zeros((M, Q, U))
    eta = np.zeros((P, L, U))
    '''Latent Variables initialization '''
    for u in range(0, U):
        tau[:, :, u] = np.random.multinomial(1, alpha[u, :], M)
        eta[:, :, u] = np.random.multinomial(1, beta[u, :], P)

    delta = np.ones((M, P, U))
    delta[X > 0] = 0

    # To store the lower bound at each iteration of the algorithm
    low_bound = np.zeros(max_iter)
    for i in range(0, max_iter):
        print(i)

        ''' VE- Step '''
        q_ij = np.zeros(
            shape=(M, P, U))  # empty array that will be used for the estimation of delta
        # empty array that will be used for the estimation of tau
        mat_R = np.zeros(shape=(M, Q, U))
        # empty array that will be used for the estimation of eta
        mat_S = np.zeros(shape=(P, L, U))

        tau[tau < 1e-16] = 1e-16  # transform zeros in  1e-16
        eta[eta < 1e-16] = 1e-16

        '''Delta Estimation'''
        '''Lines 309 and 310 are refferring to Eq. 27 and 28 of the paper'''
        for e in range(0, e_it):
            for u in range(0, U):
                q_ij[:, :, u] = pi[u, 0] * (X[:, :, u] == 0) * np.exp(
                    -np.matmul(np.matmul(tau[:, :, u], np.log(Lambda)), np.transpose(eta[:, :, u])) * X[:, :,
                                                                                                      u] + np.matmul(
                        np.matmul(tau[:, :, u], Lambda), np.transpose(eta[:, :, u])) + np.log(
                        factorial(X[:, :, u])) - np.log(1 - pi[u, 0]))
                delta[:, :, u] = q_ij[:, :, u] / (1 + q_ij[:, :, u])
            delta[(X > 0)] = 0
            delta[delta < 1e-16] = 1e-16

            ''' Tau Estimation '''
            '''Lines from 319 to 322 are refferring to Eq. 32 and 33 of the paper'''

            for u in range(0, U):
                mat_R[:, :, u] = np.matmul(np.matmul((1 - delta[:, :, u]) * X[:, :, u], eta[:, :, u]),
                                           np.transpose(np.log(
                                               Lambda))) - np.matmul(np.matmul((1 - delta[:, :, u]), eta[:, :, u]),
                                                                     np.transpose(Lambda)) + np.log(alpha[u, :])
                z_q = logsumexp(mat_R[:, :, u], axis=1)
                log_r_iq = mat_R[:, :, u] - np.array([z_q, ] * Q).transpose()
                tau[:, :, u] = np.exp(log_r_iq)

            ''' Eta Estimation '''
            '''Lines from 328 to 3331 are refferring to Eq. 40 and 41 of the paper'''

            for u in range(0, U):
                mat_S[:, :, u] = np.matmul(np.matmul(np.transpose((1 - delta[:, :, u]) * X[:, :, u]), tau[:, :, u]),
                                           np.log(
                                               Lambda)) - np.matmul(
                    np.matmul(np.transpose((1 - delta[:, :, u])), tau[:, :, u]), Lambda) + np.log(beta[u, :])
                w_l = logsumexp(mat_S[:, :, u], axis=1)
                log_s_jl = mat_S[:, :, u] - np.array([w_l, ] * L).transpose()
                eta[:, :, u] = np.exp(log_s_jl)

        tau[tau < 1e-16] = 1e-16
        eta[eta < 1e-16] = 1e-16
        pi[pi < 1e-16] = 1e-16
        alpha[alpha < 1e-16] = 1e-16
        beta[beta < 1e-16] = 1e-16
        Lambda[Lambda < 1e-16] = 1e-16

        ''' M - Step: Mixture and Sparsity parameters '''

        f_pi = MultAlpha(2, 200, 200,2)
        f_alpha = MultAlpha(Q, 200, 200, Q)
        f_beta = MultAlpha(L, 200, 200, L)

        i_inf_10 = i < 10
        jit_root_path = './traced_func/gpu' if tests["gpu"] else './traced_func'

        # latent_var: multinomial latent variables tau, eta or delta, with respectively parameters alpha, beta or pi
        # network: neural net function
        # axis_cluster:
        #     U: number of time instant, 3rd dimension of X
        #     L: number of column clusters, corresponding to the number of columns of eta
        conf = {
            'pi': {
                'name': 'pi',
                'network': f_pi,
                'axis_cluster': None,
                'latent_var': delta,
                'lr': 1e-4,
                'epochs': 2000,
                'pi_ops': True,
                'i_inf_10': i_inf_10,
                'jit': jit_root_path,
                'gpu': '0' if tests["gpu"] else None
            },
            'alpha': {
                'name': 'alpha',
                'network': f_alpha,
                'axis_cluster': Q,
                'latent_var': tau,
                'lr': 1e-4,
                'epochs': 2000,
                'pi_ops': False,
                'jit': jit_root_path,
                'gpu': '1' if tests["gpu"] else None
            },
            'beta': {
                'name': 'beta',
                'network': f_beta,
                'axis_cluster': L,
                'latent_var': eta,
                'lr': 1e-4,
                'epochs': 2000,
                'pi_ops': False,
                'jit': jit_root_path,
                'gpu': '2' if tests["gpu"] else None
            }
        }

        if tests["do_jit_saving"]:
            do_jit_saving_op(conf)

        if tests['multiprocessing']:
            ''' Multiprocessing senario'''

            '''prepare args for multiprocessing'''
            args = [conf["pi"],
                    conf["alpha"],
                    conf["beta"]]

            ## do a non-blocking map, then extract the results from the iterator
            result = Pool(3).imap(wrap_everything, args, [tests['do_jit'], tests['do_jit'], tests['do_jit']])
            result = list(result)

            '''getting results from multiprocessing'''
            res_pi = result[0]
            res_alpha = result[1]
            res_beta = result[2]

        else:
            ''' Sequential senario '''
            res_pi = wrap_everything(conf["pi"], tests['do_jit'])
            res_alpha = wrap_everything(conf["alpha"], tests['do_jit'])
            res_beta = wrap_everything(conf["beta"], tests['do_jit'])

        '''M - Step : Pi:'''

        # Parameter updatings:
        pi = res_pi[0] if res_pi[0] is not None else pi
        store_l_pi = res_pi[1]
        beta = res_beta[0]
        store_l_alpha = res_alpha[1]
        alpha = res_alpha[0]
        store_l_beta = res_beta[1]

        ''' M - Step : Lambda '''
        Lambda = np.zeros((Q, L))
        X_ql = np.zeros((Q, L))
        den1 = np.zeros((Q, L))
        dell = np.zeros((Q, L))
        den2 = np.zeros((Q, L))
        X_delta = np.zeros((M, P, U))
        pi[pi < 1e-16] = 1e-16
        Lambda[Lambda < 1e-16] = 1e-16
        '''The following loop is used to get the estimation of Lambda, it refers to Eq. 45 of the article '''
        for u in range(0, U):
            X_delta[:, :, u] = X[:, :, u] * delta[:, :, u]
            X_ql += np.matmul(np.transpose(tau[:, :, u]),
                              np.matmul(X[:, :, u], eta[:, :, u]))
            dell += np.matmul(np.transpose(tau[:, :, u]),
                              np.matmul(X_delta[:, :, u], eta[:, :, u]))
            den1 += np.matmul(np.sum(tau[:, :, u], axis=0)[..., None],
                              np.sum(eta[:, :, u], axis=0).reshape(1, L))
            den2 += np.matmul(np.transpose(tau[:, :, u]),
                              np.matmul(delta[:, :, u], eta[:, :, u]))

        Lambda = (X_ql - dell) / (den1 - den2)

        pi[pi < 1e-16] = 1e-16
        alpha[alpha < 1e-16] = 1e-16
        beta[beta < 1e-16] = 1e-16
        Lambda[Lambda < 1e-16] = 1e-16

        ''' Lower Bound Computation '''
        ''' the following loop is used to compute the lower bound (Eq. 42 of the article) that we want to maximize at every iteration of the algorithm '''
        p_x1 = p_x2 = p_x2b = p_x3 = p_delta = p_tau = p_eta = ent_tau = p_eta = ent_eta = ent_delta = p_tau = 0
        for u in range(0, U):
            p_x1 += np.sum(delta[:, :, u][X[:, :, u] == 0] * np.log(pi[u, 0]))
            p_x2 += np.sum(X[:, :, u] * (1 - delta[:, :, u]) * np.matmul(np.matmul(tau[:, :, u], np.log(
                Lambda)), np.transpose(eta[:, :, u]))) + np.sum((1 - delta[:, :, u]) * np.log(1 - pi[u, 0]))
            p_x2b += np.sum(np.matmul(np.matmul(tau[:, :, u], Lambda),
                                      np.transpose(eta[:, :, u])) * (1 - delta[:, :, u]))
            # p_x3 += np.sum((1-delta[:,:,u])*np.log(factorial(X[:,:,u]))) #it's commented because it can give numerical issues due to the log of the factorial
            p_tau += np.sum(np.matmul(tau[:, :, u], np.log(alpha[u, :])))
            ent_tau += np.sum(tau[:, :, u] * np.log(tau[:, :, u]))
            p_eta += np.sum(np.matmul(eta[:, :, u], np.log(beta[u, :])))
            ent_eta += np.sum(eta[:, :, u] * np.log(eta[:, :, u]))
            ent_delta += np.sum(delta[:, :, u] * np.log(delta[:, :, u]) +
                                (1 - delta[:, :, u]) * np.log(1 - delta[:, :, u]))

        low_bound[i] = p_x1 + p_x2 - p_x2b + p_tau + \
                       p_eta - ent_eta - ent_tau - ent_delta

    return store_l_alpha, store_l_beta, store_l_pi, tau, eta, delta, alpha, beta, pi, low_bound, Lambda, p_x1, p_x2, p_x2b, p_x3, p_tau, ent_tau, p_eta, ent_eta, ent_delta

# def main():
#     start_time = time.time()
#     X = np.load("./datas/X.npy")
#     Q = 3
#     L = 2
#     max_iter = 10
#     alpha_res = np.load("./datas/alpha_res.npy")
#     beta_res = np.load("./datas/beta_res.npy")
#     Lambda_init = np.load("./datas/Lambda_init.npy")
#     pi_mat = np.load("./datas/pi_mat.npy")
#
#     Lambda_init = Lambda_init[49, :]
#     stream_DLBM(X, Q, L, max_iter, alpha_res, beta_res, Lambda_init, pi_mat)
#     print("--- %s seconds total execution ---" % (time.time() - start_time))
#
#
# if __name__ == '__main__':
#     main()

# if you want to run the script from python or kernprof you must save datas from Script_Exp4.R
# by running it with numpy$save lines decommented
# then decomment def main in this file and then run :
# python3 StreamdLBM_main.py > test.txt

# to run kernprof you need to set @profile decorator on each functions
# kernprof -u 1e-3 -lvz -o junk.lprof StreamdLBM_main.py > ./profiling_tests/res/mac_cpu/
