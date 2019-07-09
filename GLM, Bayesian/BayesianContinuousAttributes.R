
vec <- c(95, 85, 90)
mean(vec)
sd (vec)
atr <- 95
const <- 1.0 / (sqrt (2*pi) * sd(vec))
exp.value <- exp((-((atr - mean(vec))^2) / (2 * sd(vec)^2))) 
probability <- const * exp.value
probability
class.prob <- 3.0 / 10
class.pred <- probability * class.prob
class.pred

vec <- c(125, 100, 70, 120, 60, 220, 75)
mean(vec)
sd (vec)
atr <- 95
const <- 1.0 / (sqrt (2*pi) * sd(vec))
exp.value <- exp((-((atr - mean(vec))^2) / (2 * sd(vec)^2)))
probability <- const * exp.value
probability
class.prob <- 7.0 / 10
class.pred <- probability * class.prob
class.pred
