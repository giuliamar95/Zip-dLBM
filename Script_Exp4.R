library(reticulate) #pacakage to call python script from R
# #Sys.setenv("r-reticulate" =  path.expand("~/miniconda3/envs/r-reticulate"))
# Sys.setenv("r-reticulate" =  path.expand("~/anaconda3/envs/r-reticulate"))
# Sys.setenv(RETICULATE_PYTHON=path.expand("~/anaconda3/envs/r-reticulate/bin/python"))
# RETICULATE_PYTHON=path.expand("~/anaconda3/envs/r-reticulate/bin/python")
# 
# 
# #use_python(path.expand("~/miniconda3/envs/r-reticulate/bin/python"))
# use_python(path.expand("~/anaconda3/envs/r-reticulate/bin/python"))
# 
# #use_condaenv(path.expand("~/miniconda3/envs/r-reticulate"))
# use_condaenv(path.expand("~/anaconda3/envs/r-reticulate"))
# 
# py_config()
library(reticulate)
Sys.setenv(RETICULATE_PYTHON="/Users/gmarchel/opt/anaconda3/envs/r-reticulate/bin/python")
RETICULATE_PYTHON="/Users/gmarchel/opt/anaconda3/envs/r-reticulate/bin/python"
Sys.setenv("r-reticulate" =  "/Users/gmarchel/opt/anaconda3/envs/r-reticulate")


use_python("/Users/gmarchel/opt/anaconda3/envs/r-reticulate/bin/python")
use_condaenv("/Users/gmarchel/opt/anaconda3/envs/r-reticulate")

py_config()
#import Python package in R enviroment
pandas <- import("pandas") 
torch <- reticulate::import("torch")
numpy <- import("numpy")
scipy <- import("scipy")
sklearn <- import("sklearn")
matplotlib <- import("matplotlib")
line_profiler <- import("line_profiler")
source("SimulateData_ZIP.R")
source_python("StreamdLBM_main.py") ##Main function!
source_python("LBM_ZIP.py")
M = 600
P = 400
U = 50


data = SimulateData_ZIP(M  = 600, P = 400, U = 50, "easy") #Simulate data function (Review not needed)

X = data$X

Q = 3L
L = 2L
max_iter = 20L

#Data initialization 

a_1 = runif(Q)
b_1 = runif(L)
alpha_init_1 = a_1/sum(a_1)
beta_init_1 = b_1/sum(b_1)
Lambda_init = list()
Lambda_init[[1]] = matrix(sample.int(Q*L+1, Q*L, replace = FALSE), Q,L, byrow = T)
alpha_res = matrix(NA, U, Q)
beta_res = matrix(NA, U, L)
pi_res = rep(NA, U)
pi_1 = 0.8

#LBM_ZIP is a python function for parameter initialization - Review not needed
a = Sys.time()
for (u in 1:U) {
  if(u==1){
    out_cascata = LBM_ZIP(X[,,1], as.integer(Q),  as.integer(L), max_iter, alpha_init_1,beta_init_1, Lambda_init[[1]], pi_1)
    alpha_res[1,] = out_cascata[[4]]
    beta_res[1,] = out_cascata[[5]]
    Lambda_init[[1]] = out_cascata[[8]]
    pi_res[1] = out_cascata[[6]]
  }else{
    out_cascata = LBM_ZIP(X[,,u], as.integer(Q), as.integer(L), max_iter, alpha_res[u-1,], beta_res[u-1,], Lambda_init[[u-1]], pi_res[u-1])
    alpha_res[u,] = out_cascata[[4]]
    beta_res[u,] = out_cascata[[5]]
    pi_res[u] = out_cascata[[6]]
    Lambda_init[[u]] = out_cascata[[8]]
  }
}

b = Sys.time()
print(b-a)

#to plot the results of the initialization :
# matplot(alpha_res, type = "l") 
# matplot(beta_res, type = "l")
# matplot(pi_res, type = "l")
pi_mat = matrix(NA, U, 2)
pi_mat[,1] = pi_res
pi_mat[,2] = 1-pi_res
max_iter = 10L

#MAIN FUNCTION: Written in Python using Pytorch - Review needed - 
# 54 min on to run on MacBook Pro 2020 2,3 GHz Quad-Core Intel Core i7, 16 GB RAM
a = Sys.time()
# numpy$save("./datas/X.npy", r_to_py(X))
# numpy$save("./datas/Q.npy", r_to_py(Q))
# numpy$save("./datas/L.npy", r_to_py(L))
# numpy$save("./datas/max_iter.npy", r_to_py(max_iter))
# numpy$save("./datas/alpha_res.npy", r_to_py(alpha_res))
# numpy$save("./datas/beta_res.npy", r_to_py(beta_res))
# numpy$save("./datas/Lambda_init.npy", r_to_py(Lambda_init))
# numpy$save("./datas/pi_mat.npy", r_to_py(pi_mat))
out = stream_DLBM(X, Q, L, max_iter, alpha_res, beta_res, Lambda_init[[U]], pi_mat)
b = Sys.time()
print(b-a)


#From line 80 to 90 : we recall the parameters estimated through the Stream_DLBM python function. Tuple in python -> list in R
store_l_alpha = out[[1]] 
store_l_beta = out[[2]]
store_l_pi = out[[3]]
tau = out[[4]]
eta =out[[5]]
delta = out[[6]]
alpha = out[[7]]
beta = out[[8]]
pi = out[[9]]
lower_bound =out[[10]]
Lambda =out[[11]]

save(out, data, alpha_res, beta_res,Lambda_init, pi_mat, file = "Exp-3IAtechpool.Rdata") #To save the results

#Plots of the results:
par(mfrow = c(1,1))
plot.default(lower_bound, type = "b", col = 2, main = "Lower Bound", xlab = "Iterations", ylab = "values")

par(mfrow = c(1,3))
matplot(data$alpha, type = "l", main = "True alpha", xlab = "Time (t)", ylab = " ", lwd=3)
matplot(alpha_res, type = "l", main = "Alpha Initialization", xlab = "Time (t)", ylab = " ", lwd=3)
matplot(alpha, type = "l", main = "Estimated alpha", xlab = "Time (t)", ylab = " ", lwd=3)


matplot(data$beta, type = "l", main = "True beta", xlab = "Time (t)", ylab = " ",lwd=3)
matplot(beta_res, type = "l", main = "Beta after Init", xlab = "Time (t)", ylab = " ", lwd=3)
matplot(beta, type = "l", main = "Estimated beta", xlab = "Time (t)", ylab = " ", lwd=3)

matplot(data$pi[,1], type = "l", main = "True pi", xlab = "Time (t)", ylab = " ", col = "blueviolet", lwd=3)
matplot(pi[,1], type = "l", main = "Estimated pi", xlab = "Time (t)", ylab = " ", col = "blueviolet", lwd=3)

par(mfrow = c(1,2))
image(data$Lambda)
image(Lambda)

