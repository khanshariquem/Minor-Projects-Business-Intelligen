@author Christos Argyropoulos

# This code illustrates a simple use of Bayesian Regression
# using a simple, non-informative, uniformly distributed priors


## Annette Dobson (1990) "An Introduction to Generalized Linear Models".
## Page 9: Plant Weight Data.
# Create a toy linear regression example
# (straight from R's lm help file)
ctl <- c(4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14)
trt <- c(4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69)
group <- gl(2, 10, 20, labels = c("Ctl","Trt"))
weight <- c(ctl, trt)
lmfit <- lm(weight ~ group)


