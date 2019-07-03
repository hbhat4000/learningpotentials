rm(list=ls(all=TRUE))
library(sindyr)


y = read.csv('~/Box/tfcode/rnn/modelpot1V.csv',header=FALSE)
y = as.matrix(y)
r = read.csv('~/Box/tfcode/rnn/modelpot1r.csv',header=FALSE)
r = as.matrix(r)
feat = cbind(1, r, r^2, r^3)
feat = as.matrix(feat)
r[2,1] - r[1,1] -> dr

test = sindy(xs=r, dx=y, dt=dr, Theta=feat, lambda=0.04)
print(test)
