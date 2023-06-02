#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:04:29 2022

@author: Giulia Marchello, Benjamin Navet, Marco Corneli, Charles Bouveyron
"""

import numpy as np

from scipy.special import logsumexp
#import AllFunctions
#from AllFunctions import *
import time
#from Data_Sim import *
from scipy.special import factorial
import pandas as pd


def LBM_ZIP(X, Q, L,max_iter, alpha_init, beta_init, Lambda_init, pi_init):
    alpha = np.copy(alpha_init)
    beta = np.copy(beta_init)
    Lambda = np.copy(Lambda_init)
    pi = np.copy(pi_init)
    M = X.shape[0]
    P = X.shape[1]
    '''Data initialization '''
    delta = np.ones((M,P))
    delta[X>0]= 0 
    tau = np.random.multinomial(1, alpha, M)
    eta = np.random.multinomial(1, beta, P)
    # 
    low_bound = np.zeros((max_iter))
    e_it = 5

    pi = np.mean(delta)
    tau[tau<1e-16] = 1e-16
    eta[eta<1e-16] = 1e-16
    
    
      
    alpha[alpha<1e-16] = 1e-16
    beta[beta<1e-16] = 1e-16
    Lambda[Lambda<1e-16] = 1e-16
    for i in range(0,max_iter):
      #print(np.mean(delta))
      ''' E - Step '''
        
      '''Delta Estimation'''
      q_ij = pi*(X==0)*np.exp(-np.matmul(np.matmul(tau, np.log(Lambda)), np.transpose(eta))*X +np.matmul(np.matmul(tau, Lambda), np.transpose(eta)) + np.log(factorial(X)) -np.log(1-pi))
      delta = q_ij/(1+q_ij)
      delta[(X>0)]=0 ##Corect! Checked!
      delta[delta<1e-16] = 1e-16
        
      ''' VE- Step '''
        
      ''' Tau Estimation '''
      
        
      tau[tau<1e-16] = 1e-16
      eta[eta<1e-16] = 1e-16
      
      mat_R = np.matmul(np.matmul((1-delta)*X, eta), np.transpose(np.log(Lambda))) - np.matmul(np.matmul((1-delta),eta), np.transpose(Lambda)) + np.log(alpha)
      z_q = logsumexp(mat_R, axis = 1).reshape(-1,1)
      log_r_iq = mat_R - z_q
      tau = np.exp(log_r_iq)
      
      ''' Eta Estimation '''
      mat_S = np.matmul(np.matmul(np.transpose((1-delta)*X),tau),np.log(Lambda)) -np.matmul(np.matmul(np.transpose((1-delta)),tau),Lambda)+np.log(beta) 
      w_l = logsumexp(mat_S, axis =1).reshape(-1,1)
      log_s_jl =  mat_S - w_l
      eta = np.exp(log_s_jl)
      
      tau[tau<1e-16] = 1e-16
      eta[eta<1e-16] = 1e-16
      # p_x1 = np.sum(delta*np.log(pi)+(1-delta)*np.log(1-pi))
      # p_x2 = np.sum(X*(1-delta)*np.matmul(np.matmul(tau, np.log(Lambda)),np.transpose(eta)))
      # p_x2b = np.sum(np.matmul(np.matmul(tau,Lambda),np.transpose(eta))*(1-delta))
      # p_x3 = np.sum((1-delta)*np.log(factorial(X)))
      # p_delta =np.sum(delta*np.log(pi)+(1-delta)*np.log(1-pi))
      # p_tau = np.sum(np.matmul(tau, np.log(alpha/np.sum(tau, axis = 0))))
      # p_eta = np.sum(np.matmul(eta, np.log(beta/np.sum(eta, axis = 0))))                  
      # ent_delta = np.sum(delta*np.log(delta)+(1-delta)*np.log(1-delta))
      # low_bound_ =  p_x1 +p_x2 - p_x2b - p_x3 + p_delta+ p_tau + p_eta 
      # 
      # print("E-step ", low_bound_)
      
         
      ''' M - Step : Lambda '''
      X_delta = X*delta
      X_ql = np.matmul(np.transpose(tau),np.matmul(X,eta))
      dell = np.matmul(np.transpose(tau),np.matmul(X_delta,eta))
      den = np.matmul(np.transpose(tau),np.matmul((1-delta), eta))
      Lambda = (X_ql - dell)/den
      '''Alpha, Beta:'''
    
      #print("alpha:", alpha)
      alpha = np.mean(tau, axis =0)  
      beta = np.mean(eta, axis =0)  
      # print("beta:", beta)
      pi = np.mean(delta)
      
      alpha[alpha<1e-16] = 1e-16
      beta[beta<1e-16] = 1e-16
      Lambda[Lambda<1e-16] = 1e-16
   
      ''' Lower Bound Computation '''
      
      p_x1 = np.sum(delta[X==0]*np.log(pi))
      p_x2 = np.sum(X*(1-delta)*np.matmul(np.matmul(tau, np.log(Lambda)),np.transpose(eta))) + np.sum((1-delta)*np.log(1-pi))
      p_x2b = np.sum(np.matmul(np.matmul(tau,Lambda),np.transpose(eta))*(1-delta))
      p_x3 = np.sum((1-delta)*np.log(factorial(X)))
      p_tau = np.sum(np.matmul(tau, np.log(alpha)))
      p_eta = np.sum(np.matmul(eta, np.log(beta)))                  
      ent_tau = np.sum(tau*np.log(tau))
      ent_eta = np.sum(eta*np.log(eta))
      ent_delta = np.sum(delta*np.log(delta)+(1-delta)*np.log(1-delta))

      low_bound[i]  =  p_x1 + p_x2  - p_x2b  - p_x3 + p_tau + p_eta - ent_eta - ent_tau - ent_delta

      #print("M-Step:", low_bound[i]) 
      
      
    crit = low_bound[i] - (Q-1)/2*np.log(M) - (L-1)/2*np.log(P) - (Q*L)/2*np.log(M*P) - 1/2*np.log(M*P) 
    #return store_l_alpha, store_l_beta, store_l_pi, tau, eta, delta, alpha, beta, pi, low_bound, Lambda
    return tau, eta, delta, alpha, beta, pi, low_bound, Lambda, crit

 
