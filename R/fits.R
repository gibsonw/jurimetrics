#' Fits the best model from classes ARIMA, ETS, TBATS and NNETAR.
#'
#' @param x A vector or ts object.
#' @param train The (initial) percentage of the time series to be used to train the models. Must be \code{0 < train < 1}.
#' @param steps Number of steps to forecast. If \code{NULL}, uses the number of points not used in training (testing points). Can't be less than the number of testing points.
#' @param max.points Limits the maximum number of points to be used in modeling. Uses the first \code{max.points} points of the series.
#' @param show.main.graph Logical. Should the main graphic (with the final model) be displayed?
#' @param show.sec.graph Logical. Should the secondary graphics (with the training models) be displayed?
#' @param show.value Logical. Should the values be displayed?
#' @param PI Prediction Interval used in nnar models. May take long time processing.
#' @param theme_doj Logical. Should the theme of Decades Of Jurimetrics be used?
#' @return \code{$fcast} Predicted time series using the model that minimizes the forecasting mean square error.
#' @return \code{$mse.pred} Mean squared error of prediction. Used to decide the best model.
#' @return \code{$best.model} Model that minimizes the mean squared error of prediction.
#' @return \code{$runtime} Running time.
#' @import fpp2
#' @references
#' Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting: principles and practice, 2nd edition, OTexts: Melbourne, Australia. \href{https://otexts.com/fpp2}{otexts.com/fpp2}.
#'
#' \url{https://robjhyndman.com/hyndsight/nnetar-prediction-intervals/}
#'
#' \url{https://robjhyndman.com/talks/Google-Oct2015-part1.pdf}
#'
#' Zabala, F. J. and Silveira, F. F. (2019). Decades of Jurimetrics. \url{https://arxiv.org/abs/2001.00476}#'
#' @examples
#' library(jurimetrics)
#'
#' fits(livestock)
#' fits(livestock, theme.doj = T)
#' fits(livestock, show.main.graph = F, show.sec.graph = T, show.value = F)
#'
#' fits(h02, .9)
#'
#' fits(gas)
#'
#' data('tjmg_year')
#' y1 <- ts(tjmg_year$count, start = c(2000,1), frequency = 1)
#' fits(y1)
#'
#' data(tjrs_year_month)
#' y2 <- ts(tjrs_year_month$count, start = c(2000,1), frequency = 12)
#' fits(y2, train = 0.8, steps = 24)
#' @export




fnc_eval_err <- 
  function(y_pred, y, fnc='Todos') {
  
  result <- data.frame(mse      =  mean((y_pred - y)^2),
                       rmse     =  sqrt(mean((y_pred - y)^2)),
                       rsq      =  1-(sum((y_pred - y)^2) / sum((mean(y) - y)^2)),
                       abse     =  sum(abs(y_pred - y)),
                       abse_avg =  sum(abs(y_pred - y)) / length(y)
             )
  
  
  if(fnc == 'mse'){
    #mean((d$prediction-d$y)^2)
    result <- result$mse
    }
  else if(fnc == 'rmse'){
    "
    Podemos pensar nela como sendo uma medida análoga ao desvio padrão.
    A medida RMSE tem a mesma unidade que os valores de y.
    RMSE é uma boa medida, porque geralmente ela representa explicitamente o que vários métodos tendem a minimizar
    "
    #sqrt(mean((d$prediction-d$y)^2))
    result <- result$rmse
  }
  else if(fnc == 'R-Squared'){
    "
    Ele é definido como 1.0 menos o quanto o modelo tem de de variância
    inexplicada, representando uma medida relativa a um modelo nulo que usa
    somente a média de y como o preditor
    R-squared é adimensional, e o melhor valor possível para o R-squared é 1.0(valores pequenos ou negativos de R2 não são bons sinais).
    "
    #1-sum((d$prediction-d$y)^2)/sum((mean(d$y)-d$y)^2)
    result <- result$rsq
  }
  else if(fnc == 'abs_err'){
    "
    Em muitas aplicações (especialmente aquelas que envolvem na resposta quantidade de dinheiro)    
    "
    # sum(abs(y_pred - y))
    result <- result$abse
  }
  else if(fnc == 'abs_err_avg'){
    "
    Em muitas aplicações (especialmente aquelas que envolvem na resposta quantidade de dinheiro)    
    "
    # sum(abs(y_pred - y)) / length(y)
    result <- result$abse_avg
  }
    
  return (result)
}


fits <- function(x, train = 0.8,
                 steps = NULL,
                 max.points = 500,
                 show.main.graph = T,
                 show.sec.graph = F,
                 show.value = T,
                 PI = F,
                 theme.doj = F){
  
  ini <- Sys.time()
  
  # filtering max.points
  n0 <- length(x)
  if(n0 > max.points)
    {x <- x[(n0-max.points+1):n0]}
  
  # train-test
  n <- length(x)
  i <- ceiling(train*n)
  xTrain <- 1:i
  xTest <- (i+1):n
  
  # models
  fit.aa <- forecast::auto.arima(x[xTrain])
  fit.ets <- forecast::ets(x[xTrain])
  fit.tb <- forecast::tbats(x[xTrain])
  set.seed(1); fit.nn <- forecast::nnetar(x[xTrain])
  
  # forecast
  if(is.null(steps)) 
    {steps <- length(xTest)}
      fcast.aa <- forecast::forecast(fit.aa, h=steps)
      fcast.ets <- forecast::forecast(fit.ets, h=steps)
      fcast.tb <- forecast::forecast(fit.tb, h=steps)
      fcast.nn <- forecast::forecast(fit.nn, h=steps, PI = PI)
  
  # akaike information criteria
  aic <- cbind(aa  = c(aic=fit.aa$aic, aicc=fit.aa$aicc, bic=fit.aa$bic),
               ets = c(aic=fit.ets$aic, aicc=fit.ets$aicc, bic=fit.ets$bic),
               tb  = c(aic=fit.tb$AIC, aicc=NA, bic=NA),
               nn  = c(aic=NA, aicc=NA, bic=NA))
  
  # mean squared error (residuals)
  mse.fit <- data.frame(mse.fit.aa = mean(residuals(fit.aa)^2),
                        mse.fit.ets = mean(residuals(fit.ets)^2),
                        mse.fit.tb = mean(residuals(fit.tb)^2),
                        mse.fit.nn = mean(residuals(fit.nn)^2))
  

  # retorna odas medidas de erro
  if(train != 1){
    all_fnc_err <- data.frame(pred.aa  =  fnc_eval_err(fcast.aa$mean[1:length(xTest)] , x[xTest]),
                          pred.ets =  fnc_eval_err(fcast.ets$mean[1:length(xTest)], x[xTest]),
                          pred.tb  =  fnc_eval_err(fcast.tb$mean[1:length(xTest)] , x[xTest]),
                          pred.nn  =  fnc_eval_err(fcast.nn$mean[1:length(xTest)] , x[xTest]))
  }

  # mean squared error (forecast)
  if(train != 1){
    mse.pred <- data.frame(mse.pred.aa  =  all_fnc_err$pred.aa.mse,
                           mse.pred.ets =  all_fnc_err$pred.ets.mse,
                           mse.pred.tb  =  all_fnc_err$pred.tb.mse,
                           mse.pred.nn  =  all_fnc_err$pred.nn.mse)
  }
  
  # fitting best model based on mse.pred
  bestModel <- which.min(mse.pred)
  
  if(bestModel == 1){
    best.model <- 'arima'
    fit <- forecast::auto.arima(x)
    fcast <- forecast::forecast(fit, h = steps)
  }
  else if(bestModel == 2){
    best.model <- 'ets'
    fit <- forecast::ets(x)
    fcast <- forecast::forecast(fit, h = steps)
  }
  else if(bestModel == 3){
    best.model <- 'tbats'
    fit <- forecast::tbats(x)
    fcast <- forecast::forecast(fit, h = steps)
  }
  else if(bestModel == 4){
    best.model <- 'nnetar'
    set.seed(1); fit <- forecast::nnetar(x)
    fcast <- forecast::forecast(fit, h = steps, PI = PI)
  }
  
  # train/test plots
  if(show.sec.graph){
    par(mfrow=c(2,2))
    plot(fcast.aa); points(c(rep(NA,i), x[xTest]))
    plot(fcast.ets); points(c(rep(NA,i), x[xTest]))
    plot(fcast.tb); points(c(rep(NA,i), x[xTest]))
    plot(fcast.nn); points(c(rep(NA,i), x[xTest]))
  }
  
  # main plot (best model)
  if(show.main.graph){
    
    if(!theme.doj){
      print(ggplot2::autoplot(fcast))
    }
    
    if(theme.doj){
      print(ggplot2::autoplot(fcast) +
              jurimetrics::theme_doj())
    }
  }
  
  # mean squared error (best fit residuals)
  # mse.fit.best <- mean(residuals(fit)^2)

  # presenting results
  if(show.value){
    return(list(fcast = fcast,
                mse.pred = mse.pred,
                all_fnc_err = all_fnc_err,
                aic = aic,
                best.model = best.model
                )
           )
  }
}
