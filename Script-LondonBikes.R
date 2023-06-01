library(readxl) 
library("xlsx")

X1 <- read_excel("./320JourneyDataExtract01Jun2022-07Jun2022.xlsx")
X2 <- read_excel("./321JourneyDataExtract08Jun2022-14Jun2022.xlsx")
X3 <- read_excel("./322JourneyDataExtract15Jun2022-21Jun2022.xlsx")
X4 <- read_excel("./323JourneyDataExtract22Jun2022-28Jun2022.xlsx")

X = do.call(rbind, mget(paste0("X", 1:4)))


library(dplyr)
Tin = "2022-06-01"
Tout = "2022-06-29"

X$dates = format(X$dates, "%Y-%m-%d-%H")
X$dates = as.POSIXct(X$`Start Date`, origin="2021-09-22")
X = arrange(X,dates)
X = dplyr::filter(X,dates>Tin&dates<Tout)


num = as.numeric(format(X$dates, format = "%H"))
X=X[which(num>=06),]
X=X[which(num<22),]
X$hour = format(X$dates, format = "%H")

X$hourID = as.numeric(as.factor(X$hour))  

sel.start = names(which(table(X$`StartStation Name`)>50))
sel.end = names(which(table(X$`EndStation Name`)>50))
ind = which((X$`EndStation Name` %in% sel.end) & (X$`StartStation Name`  %in% sel.start))
X = X[ind,]

# 
X$SId = as.numeric(as.factor(as.character(X$`StartStation Name`)))
X$EId = as.numeric(as.factor(as.character(X$`EndStation Name`)))
X$days = weekdays(X$dates)    
d = c("Monday","Tuesday", "Wednesday", "Thursday", "Friday")
X = X[which(X$days %in% d),] 


par(mfrow=c(1,1))
hist(X$dates,breaks="days", freq = T, col = 2, main = "Histogram of complete data Jun 2021", xlab = "Dates")
barplot(table(X$hour), col = 2, main = "London bikes - June 2022", xlab = "Hours")

# 
library(slam)
N = length(unique(X$SId))
P = length(unique(X$EId))
U = length(unique(X$hourID))
A = simple_sparse_zero_array(dim = c(N,P,U))
coord = dplyr::select(X,SId,EId,hourID)
for (i in 1:nrow(coord)) A[coord$SId[i],coord$EId[i],coord$hourID[i]][1] = A[coord$SId[i],coord$EId[i],coord$hourID[i]][1] + 1
D = array(0, c(N,P,U))
for (u in 1:U) {
  D[,,u] = matrix(as.vector(A[,,u]), nrow = N, ncol = P)
}

save(A, file = "London_June2022.Rdata")


max_iter = 20L
Q  = 5:9
L = 5:9
mod = expand.grid(Q=Q, L=L)

icl = rep(NA,dim(mod)[1])
out_test = list()
nsim = 1
data_sim = 10
best_mod = matrix(NA, nrow=data_sim, ncol=3)
library(raster)
s = rep(NA, U)
for (u in 1:U) {
  print(u)
  s[u] = sum(as.vector(A[,,u]))
  print("***")
}
for (n in 1:data_sim) {
  
  for(i in 1:dim(mod)[1]){
    a_1 = runif(mod[i,1])
    b_1 = runif(mod[i,2])
    alpha = rep(1/mod[i,1], mod[i,1])
    beta = rep(1/ mod[i,2], mod[i,2])
    alpha = a_1/sum(a_1)
    beta = b_1/sum(b_1)
    pi_init = runif(1)
    Lambda = matrix(sample.int(mod[i,1]*mod[i,2]+1, mod[i,1]*mod[i,2], replace = FALSE), mod[i,1],mod[i,2], byrow = T)
    
    out_test[[i]] = LBM_ZIP(D[,,4], mod[i,1], mod[i,2], max_iter, alpha, beta, Lambda, pi_init)
    print(i)
    print(out_test[[i]][[9]])
    icl[i] = out_test[[i]][[9]]
  }
  
  mod[,3] = unlist(icl)
  mod_ord = as.matrix(mod[order(mod$V3,decreasing = TRUE),])
  best_mod[n,] = as.vector(mod_ord[1,])
}
###LBM Cascade process!###
par(mfrow = c(1,1))

plot.default(unlist(icl), type = "b")
Mode <-
  function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  }

Q = Mode(best_mod[,1])
L = Mode(best_mod[,2])

a_1 = runif(Q) 
b_1 = runif(L)
alpha_init_1 = a_1/sum(a_1)
beta_init_1 = b_1/sum(b_1)
Lambda_init = list()
Lambda_init[[1]] = matrix(sample.int(Q*L+1, Q*L, replace = FALSE), Q,L, byrow = T)
alpha_res = matrix(NA, U, Q)
beta_res = matrix(NA, U, L)
pi_res = rep(NA, U)
pi_1 = 0.9


max_iter = 20L
a = Sys.time()
for (u in 1:U) {
  if(u==1){
    out_cascata = LBM_ZIP(D[,,1], as.integer(Q),  as.integer(L), max_iter, alpha_init_1,beta_init_1, Lambda_init[[1]], pi_1)
    alpha_res[1,] = out_cascata[[4]]
    beta_res[1,] = out_cascata[[5]]
    Lambda_init[[1]] = out_cascata[[8]]
    pi_res[1] = out_cascata[[6]]
  }else{
    out_cascata = LBM_ZIP(D[,,u], as.integer(Q), as.integer(L), max_iter, alpha_res[u-1,], beta_res[u-1,], Lambda_init[[u-1]], pi_res[u-1])
    alpha_res[u,] = out_cascata[[4]]
    beta_res[u,] = out_cascata[[5]]
    pi_res[u] = out_cascata[[6]]
    Lambda_init[[u]] = out_cascata[[8]]
  }
}

matplot(alpha_res, type = "l")
matplot(beta_res, type = "l")
matplot(pi_res, type = "l")

aggr_D = apply(D[,,1:3], 3, sum)

max_iter = 10L
Q_stream = 10L
L_stream = 10L

pi_mat= matrix(NA, U, 2)
pi_mat[,1] = pi_res
pi_mat[,2] = 1-pi_res

alpha_orig = array(0, c(U,Q_stream))
beta_orig = array(0, c(U,L_stream))
Lambda_orig = array(0, c(Q_stream,L_stream))

alpha_orig[,1:ncol(alpha_res)] = alpha_res

beta_orig[,1:ncol(beta_res)] = beta_res
Lambda_orig[1:nrow(Lambda_init[[U]]),1:ncol(Lambda_init[[U]])]  = Lambda_init[[U]]
write.csv(pi_mat, file = "pi_mat.csv")

a = Sys.time()
out2 = Stream_DLBM(D, Q_stream, L_stream, max_iter, alpha_orig, beta_orig, Lambda_orig, pi_mat)
b = Sys.time()
print(b-a)
store_l_alpha = out2[[1]]
store_l_beta = out2[[2]]
store_l_pi = out2[[3]]
tau = out2[[4]]
eta =out2[[5]]
delta = out2[[6]]
alpha = out2[[7]]
beta = out2[[8]]
pi = out2[[9]]
lower_bound =out2[[10]]
Lambda =out2[[11]]

par(mfrow = c(1,1))
plot.default(lower_bound, type = "b", col = 2, main = "Lower Bound", xlab = "Iterations", ylab = "values")
par(mfrow = c(1,1))
colors = c("#04F6FF","#008000","#FFB506","#FF1493","#800000", "#9400D3", "#0404F5", "#00FF00","#00BFFF","#FF4500", "#808080")

#matplot(alpha_res, type = "l", main = "True alpha", xlab = unique(X$hourID), ylab = " ", col =colors[1:6], lwd = 3)
par(mar=c(5, 4, 4, 8), xpd=TRUE)
matplot(alpha, type = "l", main = expression(paste('Estimated  ',alpha(t),sep='')), xaxt = "n", xlab = " " , ylab = " ", col =c(colors[1:6], rep("black", 4)), lty = 1:6, lwd = 4)
axis(1, 1:16 , labels = seq(6, 21, by = 1))
legend("topright", inset=c(-0.26, 0),  col = colors[1:6], cex = 0.8,
       c("Cluster 1","Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6"), lty = 1:6)
dev.copy2pdf(file = "Aestim.pdf", width = 7, height = 6)

#matplot(beta_res, type = "l", main = "True beta", xlab = unique(X$hourID), ylab = " ",col =colors[7:12], lwd = 3)
par(mar=c(5, 4, 4, 8), xpd=TRUE)
matplot(beta, type = "l", main = expression(paste('Estimated  ',beta(t),sep='')), xaxt = "n", ylab = " ", col = c(colors[7:12], rep("black", 4)),  lty = 7:12, lwd = 4)
axis(1, 1:16 , labels = seq(6, 21, by = 1))
legend("topright", inset=c(-0.26, 0), col =colors[7:12] ,cex = 0.8,
       c("Cluster 1","Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6"),  lty = 7:12)
dev.copy2pdf(file = "Bestim.pdf", width =7, height = 6)

par(mfrow = c(1,1))
par(mar=c(5, 4, 4, 4), xpd=FALSE)

matplot(pi_res, type = "l", main = expression(paste('Estimated  ',pi(t),sep='')), xlab = "Time (t)", ylab = " ", col = "blueviolet", lwd = 2)

matplot(pi[,1], type = "l",main = expression(paste('Estimated  ',pi(t),sep='')), xaxt = "n", xlab = "", ylab = " ", col = "blueviolet", lwd = 3)
axis(1, 1:16 , labels = seq(6, 21, by = 1))


dev.copy2pdf(file = "Pestim.pdf", width = 10, height = 6)

bike_loc = read_excel("./bike-points.xlsx")
library(ggplot2)
library(ggmap)
library(maps)
library(mapdata)
library(dplyr)
library(leaflet)

dim(bike_loc)
colors = c("#04F6FF","#008000","#FFB506","#FF1493","#800000", "#9400D3","#E600FF", "#0404F5", "#00FF00","#00BFFF","#FF4500", "#808080")
#colors = brewer.pal(n=10, "Paired")

for (u in 1:U) {
  pk = list()
  pl =list()
  print(u)
  for (k in 1:ncol(alpha)){
    sel = which(max.col(tau[,,u]) == k)
    names = sapply(sel,function(x) as.character(X$`StartStation Name`[which(X$SId == x)][1]))
    pk[[k]]=paste(names, sep = ", ")
  }
  
  for (l in 1:ncol(beta)){
    sel = which(max.col(eta[,,u]) == l)
    names = sapply(sel,function(x) as.character(X$`EndStation Name`[which(X$EId == x)][1]))
    pl[[l]]=paste(names, sep = ", ")
  }
  
  cl_1_row = bike_loc[which(bike_loc$name%in%pk[[1]]==TRUE),]
  cl_1_row$col = rep(colors[1], nrow(cl_1_row))
  cl_2_row = bike_loc[which(bike_loc$name%in%pk[[2]]==TRUE),]
  cl_2_row$col = rep(colors[2], nrow(cl_2_row))
  cl_3_row = bike_loc[which(bike_loc$name%in%pk[[3]]==TRUE),]
  cl_3_row$col = rep(colors[3], nrow(cl_3_row))
  cl_4_row = bike_loc[which(bike_loc$name%in%pk[[4]]==TRUE),]
  cl_4_row$col = rep(colors[4], nrow(cl_4_row))
  cl_5_row = bike_loc[which(bike_loc$name%in%pk[[5]]==TRUE),]
  cl_5_row$col = rep(colors[5], nrow(cl_5_row))
  cl_6_row = bike_loc[which(bike_loc$name%in%pk[[6]]==TRUE),]
  cl_6_row$col = rep(colors[6], nrow(cl_6_row))
  
  cl_1_col = bike_loc[which(bike_loc$name%in%pl[[1]]==TRUE),]
  cl_1_col$col = rep(colors[7], nrow(cl_1_col))
  cl_2_col = bike_loc[which(bike_loc$name%in%pl[[2]]==TRUE),]
  cl_2_col$col = rep(colors[8], nrow(cl_2_col))
  cl_3_col = bike_loc[which(bike_loc$name%in%pl[[3]]==TRUE),]
  cl_3_col$col = rep(colors[9], nrow(cl_3_col))
  cl_4_col = bike_loc[which(bike_loc$name%in%pl[[4]]==TRUE),]
  cl_4_col$col = rep(colors[10], nrow(cl_4_col))
  cl_5_col = bike_loc[which(bike_loc$name%in%pl[[5]]==TRUE),]
  cl_5_col$col = rep(colors[11], nrow(cl_5_col))
  cl_6_col = bike_loc[which(bike_loc$name%in%pl[[6]]==TRUE),]
  cl_6_col$col = rep(colors[12], nrow(cl_6_col))
  
  df_row=rbind(cl_1_row, cl_2_row, cl_3_row, cl_4_row, cl_5_row, cl_6_row)
  df_col=rbind(cl_1_col, cl_2_col, cl_3_col, cl_4_col, cl_5_col, cl_6_col)
  
  
  
  m1<-leaflet()%>%addTiles()%>%addCircleMarkers(radius = 1.7, color = df_row$col, opacity = 0.8, lng =as.numeric(c(cl_1_row$longitude, cl_2_row$longitude, cl_3_row$longitude, cl_4_row$longitude, cl_5_row$longitude,  cl_6_row$longitude)), 
                                                lat = as.numeric(c(cl_1_row$latitude, cl_2_row$latitude, cl_3_row$latitude, cl_4_row$latitude, cl_5_row$latitude, cl_6_row$latitude ))) %>%
    addLegend(position = "bottomright",colors = colors[1:6], labels = c("Cluster 1","Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6")) %>%
    setView(lng = -0.12, lat = 51.5, zoom = 12.49) 
  
  
  
  
  m2<-leaflet()%>%addTiles()%>%addCircleMarkers(radius = 1.7, color = df_col$col, opacity = 0.8, lng =as.numeric(c(cl_1_col$longitude, cl_2_col$longitude, cl_3_col$longitude, cl_4_col$longitude, cl_5_col$longitude, cl_6_col$longitude)), 
                                                lat = as.numeric(c(cl_1_col$latitude, cl_2_col$latitude, cl_3_col$latitude, cl_4_col$latitude, cl_5_col$latitude, cl_6_col$latitude)))%>%
    addLegend(position = "bottomright",colors =colors[7:12], labels = c("Cluster 1","Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6")) %>%
    setView(lng = -0.12, lat = 51.5, zoom = 12.49) 
  
  
  
  saveWidget(m1, "temp.html", selfcontained = FALSE)
  webshot("temp.html", file = paste0("start_t",u,".png"),
          cliprect = "viewport")
  saveWidget(m2, "temp.html", selfcontained = FALSE)
  webshot("temp.html", file = paste0("end_t",u,".png"),
          cliprect = "viewport")
  
}

colors = c("#04F6FF","#008000","#FFB506","#FF1493","#800000", "#9400D3","#E600FF", "#0404F5", "#00FF00","#00BFFF","#FF4500", "#808080")
#colors = brewer.pal(n=10, "Paired")

for (u in 1:U) {
  pk = list()
  pl =list()
  print(u)
  for (k in 1:ncol(alpha)){
    sel = which(max.col(tau[,,u]) == k)
    names = sapply(sel,function(x) as.character(X$`StartStation Name`[which(X$SId == x)][1]))
    pk[[k]]=paste(names, sep = ", ")
  }
  
  for (l in 1:ncol(beta)){
    sel = which(max.col(eta[,,u]) == l)
    names = sapply(sel,function(x) as.character(X$`EndStation Name`[which(X$EId == x)][1]))
    pl[[l]]=paste(names, sep = ", ")
  }
  
  # cl_1_row = bike_loc[which(bike_loc$name%in%pk[[1]]==TRUE),]
  # cl_1_row$col = rep(colors[1], nrow(cl_1_row))
  cl_2_row = bike_loc[which(bike_loc$name%in%pk[[2]]==TRUE),]
  cl_2_row$col = rep(colors[2], nrow(cl_2_row))
  # cl_3_row = bike_loc[which(bike_loc$name%in%pk[[3]]==TRUE),]
  # cl_3_row$col = rep(colors[3], nrow(cl_3_row))
  # cl_4_row = bike_loc[which(bike_loc$name%in%pk[[4]]==TRUE),]
  # cl_4_row$col = rep(colors[4], nrow(cl_4_row))
  # cl_5_row = bike_loc[which(bike_loc$name%in%pk[[5]]==TRUE),]
  # cl_5_row$col = rep(colors[5], nrow(cl_5_row))
  # cl_6_row = bike_loc[which(bike_loc$name%in%pk[[6]]==TRUE),]
  # cl_6_row$col = rep(colors[6], nrow(cl_6_row))
  
  # cl_1_col = bike_loc[which(bike_loc$name%in%pl[[1]]==TRUE),]
  # cl_1_col$col = rep(colors[7], nrow(cl_1_col))
  cl_2_col = bike_loc[which(bike_loc$name%in%pl[[2]]==TRUE),]
  cl_2_col$col = rep(colors[8], nrow(cl_2_col))
  # cl_3_col = bike_loc[which(bike_loc$name%in%pl[[3]]==TRUE),]
  # cl_3_col$col = rep(colors[9], nrow(cl_3_col))
  # cl_4_col = bike_loc[which(bike_loc$name%in%pl[[4]]==TRUE),]
  # cl_4_col$col = rep(colors[10], nrow(cl_4_col))
  # cl_5_col = bike_loc[which(bike_loc$name%in%pl[[5]]==TRUE),]
  # cl_5_col$col = rep(colors[11], nrow(cl_5_col))
  # cl_6_col = bike_loc[which(bike_loc$name%in%pl[[6]]==TRUE),]
  # cl_6_col$col = rep(colors[12], nrow(cl_6_col))
  
  # df_row=rbind(cl_1_row, cl_5_row)
  # df_col=rbind(cl_4_col, cl_6_col)
  
  df_row=cl_2_row
  df_col=cl_2_col
  
  
  m1<-leaflet()%>%addTiles()%>%addCircleMarkers(radius = 1.7, color = df_row$col, opacity = 0.8, lng =as.numeric(c(cl_2_row$longitude)), 
                                                lat = as.numeric(c(cl_2_row$latitude))) %>%
    addLegend(position = "bottomright",colors =  colors[2], labels = c("Cluster 2")) %>%
    setView(lng = -0.12, lat = 51.5, zoom = 12.49) 
  
  
  
  
  m2<-leaflet()%>%addTiles()%>%addCircleMarkers(radius = 1.7, color = df_col$col, opacity = 0.8, lng =as.numeric(cl_2_col$longitude), 
                                                lat = as.numeric(cl_2_col$latitude))%>%
    addLegend(position = "bottomright",colors =colors[8], labels = "Cluster 2") %>%
    setView(lng = -0.12, lat = 51.5, zoom = 12.49) 
  
  
  
  saveWidget(m1, "temp.html", selfcontained = FALSE)
  webshot("temp.html", file = paste0("2cl_start_t",u,".png"),
          cliprect = "viewport")
  saveWidget(m2, "temp.html", selfcontained = FALSE)
  webshot("temp.html", file = paste0("2cl_end_t",u,".png"),
          cliprect = "viewport")
  
}



#histogram
for (u in 1:U) {
  h = hist(as.numeric(X$hour), breaks = U,col = colors, main = "Histogram of complete data Jun 2021", xlab = "Dates")
  hist_breaks <- h$breaks
  color_list <- rep("#33FFE0", length(hist_breaks))
  color_list[hist_breaks = u] <- "#F9ACEC"
  
  barplot(table(X$hour), breaks = U,col = color_list, main = paste0("22-09-2021, t=",u) , xlab = "Hours")
  dev.copy2pdf(file = paste0("hist_",u,".pdf"), width = 5, height = 5)
}
Lambda_ = Lambda[1:6, 1:6]
#Lambda
NewColors = c("#A6CEE3","#1F78B4","#FFB506","#33A02C","#FB9A99","#E31A1C","#FDBF6F","#FF7F00","#CAB2D6","#6A3D9A","#C3FF99","#B15928", "#C39BD3","#F7A6DD","#33EEFF","#CF33FF", "#C70039","#3498DB", "#33FFB0", "#2AAE7B", "#B2BABB", "#EDAFC4", "#EDC0AF")
j = 1:L
i = 1:Q
image( 1:(L+1),1:(Q+1),t(Lambda_)^0.3,col=colorRampPalette(c('white',NewColors[2]),alpha=TRUE)(255), axes=FALSE,
       useRaster = TRUE, xlab = " ", line = 1, ylab = " ")
text(y=seq(1+0.5,Q+0.5,1),  x=rep(0.9,6),labels = paste("D", seq(1,Q), sep = ""),par("usr")[1], pos=2, cex= 1.5, adj = 0,xpd=TRUE, col = colors[1:6])
points(y=seq(1+0.5,Q+0.5,1), x=rep(0.9,6), cex= 2.5, xpd=TRUE, pch = 16,col = colors[1:6])
text(x=seq(1+0.5,L+0.5,1),  y=rep(0.7,6),labels = paste("A", seq(1,L), sep = ""),par("usr")[1], pos=1, cex= 1.5, adj = 0,xpd=TRUE, col = colors[7:12])

points(x=seq(1+0.5,L+0.5,1), y=rep(0.8,6), cex= 2.5, xpd=TRUE, pch = 16,col = colors[7:12])

#text(seq(1+0.5,L+0.5,1), labels = paste(seq(1,L), sep = ""),par("usr")[1], pos=1, cex=1.5, adj = 0,xpd=TRUE, col = colors[7:12])
ij <- expand.grid(i=i, j=j)
text (x=ij$j+0.5, y=ij$i+0.5, paste(round(Lambda[i,j], 2), sep = " "))
abline(v=seq(1:L+1),lty=2, lwd=0.25)
abline(h=seq(1:Q+1),lty=2, lwd=0.25)
box()


dev.copy2pdf(file = "Lambda.pdf", width = 6, height = 6)



cl_1_row = bike_loc[which(bike_loc$name%in%pk[[1]]==TRUE),]
cl_1_row$col = rep("red", nrow(cl_1_row))
cl_2_row = bike_loc[which(bike_loc$name%in%pk[[2]]==TRUE),]
cl_2_row$col = rep("blue", nrow(cl_2_row))

cl_3_row = bike_loc[which(bike_loc$name%in%pk[[3]]==TRUE),]
cl_3_row$col = rep("green", nrow(cl_3_row))

cl_1_col = bike_loc[which(bike_loc$name%in%pl[[1]]==TRUE),]
cl_1_col$col = rep("red", nrow(cl_1_col))

cl_2_col = bike_loc[which(bike_loc$name%in%pl[[2]]==TRUE),]
cl_2_col$col = rep("blue", nrow(cl_2_col))

cl_3_col = bike_loc[which(bike_loc$name%in%pl[[3]]==TRUE),]
cl_3_col$col = rep("green", nrow(cl_3_col))

cl_4_col = bike_loc[which(bike_loc$name%in%pl[[4]]==TRUE),]
cl_4_col$col = rep("#04F6FF", nrow(cl_4_col))


library(leaflet)
df_row=rbind(cl_1_row, cl_2_row, cl_3_row)
df_col=rbind(cl_1_col, cl_2_col, cl_3_col, cl_4_col)


m1<-leaflet()%>%addTiles()%>%addCircleMarkers(radius = 1.5, color = df_row$col, lng =as.numeric(c(cl_1_row$longitude, cl_2_row$longitude, cl_3_row$longitude)), 
                                              lat = as.numeric(c(cl_1_row$latitude, cl_2_row$latitude, cl_3_row$latitude ))) %>%
  addLegend(position = "bottomright",colors = unique(df_row$col), labels = paste0("Cluster ", 1:length(unique(df_row$col)))) %>%   
  setView(lng = -0.12, lat = 51.5, zoom = 12.5) 

m1

m2<-leaflet()%>%addTiles()%>%addCircleMarkers(radius = 1.5, color = df_col$col, lng =as.numeric(c(cl_1_col$longitude, cl_2_col$longitude, cl_3_col$longitude, cl_4_col$longitude)), 
                                              lat = as.numeric(c(cl_1_col$latitude, cl_2_col$latitude, cl_3_col$latitude, cl_4_col$latitude )))%>%
  addLegend(position = "bottomright",colors = unique(df_col$col), labels = paste0("Cluster ", 1:length(unique(df_col$col))))


#m<- leaflet()%>%addTiles()%>%addCircleMarkers(radius = 1,color = "red", lng =as.numeric(bike_loc$longitude), lat = as.numeric(bike_loc$latitude ))
saveWidget(m1, "temp.html", selfcontained = FALSE)
webshot("temp.html", file = "start_t12.png",
        cliprect = "viewport")
saveWidget(m2, "temp.html", selfcontained = FALSE)
webshot("temp.html", file = "end_t12.png",
        cliprect = "viewport")




library(webshot)
library(htmlwidgets)
m<- leaflet()%>%addTiles()%>%addCircleMarkers(radius = 1,color = "red", lng =as.numeric(bike_loc$longitude), lat = as.numeric(bike_loc$latitude ))
saveWidget(m, "temp.html", selfcontained = FALSE)
webshot("temp.html", file = "LondonWeb.png",
        cliprect = "viewport")


