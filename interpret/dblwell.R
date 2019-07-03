rm(list=ls(all=TRUE))
library(sindyr)

mylamb = 0.5

y = read.csv('~/Box/tfcode/rnn/dblwellV.csv',header=FALSE)
y = as.matrix(y)
r = read.csv('~/Box/tfcode/rnn/dblwellr.csv',header=FALSE)
r = as.matrix(r)
feat = cbind(1, r, r^2, r^3, r^4, r^5, r^6)
feat = as.matrix(feat)
r[2,1] - r[1,1] -> dr

test = sindy(xs=r, dx=y, dt=dr, Theta=feat, lambda=mylamb)
print(test)


y = read.csv('~/Box/tfcode/rnn/dblwellVoneside.csv',header=FALSE)
y = as.matrix(y)
r = read.csv('~/Box/tfcode/rnn/dblwellroneside.csv',header=FALSE)
r = as.matrix(r)
feat = cbind(1, r, r^2, r^3, r^4, r^5, r^6)
feat = as.matrix(feat)
r[2,1] - r[1,1] -> dr

test = sindy(xs=r, dx=y, dt=dr, Theta=feat, lambda=mylamb)
print(test)


y = read.csv('~/Box/tfcode/rnn/dblwellVNEW.csv',header=FALSE)
y = as.matrix(y)
r = read.csv('~/Box/tfcode/rnn/dblwellrNEW.csv',header=FALSE)
r = as.matrix(r)
feat = cbind(1, r, r^2, r^3, r^4, r^5, r^6)
feat = as.matrix(feat)
r[2,1] - r[1,1] -> dr

test = sindy(xs=r, dx=y, dt=dr, Theta=feat, lambda=mylamb)
print(test)