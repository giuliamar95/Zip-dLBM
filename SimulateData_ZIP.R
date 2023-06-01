library(slam)
library(Rlab)
################################
# Model: Zip-dLBM
# Authors: Giulia Marchello, Benjamin Navet, Marco Corneli, Charles, Bouveyron
################################


SimulateData_ZIP <- function(M = 200, P = 300, U = 100, ...){
  
  alpha = matrix(NA, nrow = U, ncol = 3)
  alpha[,1] = log(seq(from = 0.65, to = 0.2, length = U))^2
  alpha[,2] = log(seq(from = 0.95, to = 0.4, length = U))^2
  alpha[,3] = log(seq(from = 0.35, to = 0.85, length = U))^4
  alpha = exp(alpha)/rowSums(exp(alpha))
  beta = matrix(NA, nrow = U, ncol = 2)
  beta[,1] = seq(from = 5, to = 2, length = U)
  beta[,2] = seq(from = 1, to = 6, length = U)
  beta = exp(beta)/rowSums(exp(beta))
  
  
  X = array(NA,c(M, P, U))
  A = array(0,c(M, P, U))
  
  pi = matrix(NA, nrow = U, ncol = 2)
  pi[,2] = seq(from = 5, to = 2, length = U)
  pi[,1] = seq(from = 1, to = 6, length = U)
  pi = exp(pi)/rowSums(exp(pi))
  pi = 0.8+ pi/9  
  
  Lambda = rbind(c(6, 4),
                 c(1, 2),
                 c(7, 3))
  
  Q = ncol(alpha)
  L = ncol(beta)
  Zinit = array(NA, c(M, Q, U))
  Winit = array(NA, c(P, L, U))
  Z = matrix(NA, nrow = M, ncol = U)
  W = matrix(NA, nrow = P, ncol = U)
  X = array(NA,c(M, P, U))
  A = array(0,c(M, P, U))
  
  for (u in 1:U) {
    Zinit[,,u] = t(rmultinom(M,1,alpha[u,]))
    Winit[,,u] = t(rmultinom(P,1,beta[u,]))
    Z[,u] = max.col(Zinit[,,u])
    W[,u] = max.col(Winit[,,u])
    
    A[,,u] = rbern(M*P, pi[u,1])
    for (q in 1:Q) {
      for (l in 1:L) {
        sel_z =which(Z[,u]==q)
        sel_w = which(W[,u]==l)
        X[sel_z,sel_w,u] = rpois(length(sel_z)*length(sel_w), Lambda[q,l])
      }
    }
  }
  
  X[which(A==1)]=0
  
  
  list(X=X,row_clust=Z,col_clust=W, A=A, pi = pi,alpha = alpha, beta= beta, Zinit = Zinit, Winit = Winit, Lambda = Lambda)
}


