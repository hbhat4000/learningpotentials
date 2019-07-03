rm(list=ls(all=TRUE))
library(sindyr)


y = read.csv('~/Box/tfcode/rnn/pot2Vtrain.csv',header=FALSE)
y = as.matrix(y)
r = read.csv('~/Box/tfcode/rnn/pot2rtrain.csv',header=FALSE)
r = as.matrix(r)
feat = cbind(1, r^(-1), r^(-2), r^(-3))
feat = as.matrix(feat)
r[2,1] - r[1,1] -> dr

test = sindy(xs=r, dx=y, dt=dr, Theta=feat, lambda=0.05)
print(test)

print((-0.07792 - (-0.07958)) / -0.07958)
print((-0.06911 - (-0.07958)) / -0.07958)