% likelihood functions are provided to be used by the gp.m function:
%
%   likErf         (Error function, classification, probit regression)
%   likLogistic    (Logistic,       classification, logit  regression)
%
%   likGauss       (Gaussian, regression)
%   likLaplace     (Laplacian or double exponential, regression)
%   likSech2       (Sech-square, regression)
%   likT           (Student's t, regression)
%
% The likelihood functions have three possible modes, the mode being selected
% as follows (where "lik" stands for any likelihood function in "lik/lik*.m".):
%
% 1) With one or no input arguments:          [REPORT NUMBER OF HYPERPARAMETERS]
%
%    s = lik OR s = lik(hyp)
%
% The likelihood function returns a string telling how many hyperparameters it
% expects, using the convention that "D" is the dimension of the input space.
% For example, calling "likLogistic" returns the string '0'.
%
%
% 2) With three or four input arguments:                       [PREDICTION MODE]
%
%    lp = lik(hyp, y, mu) OR [lp, ymu, ys2] = lik(hyp, y, mu, s2)
%
% This allows to evaluate the predictive distribution. Let p(y_*|f_*) be the
% likelihood of a test point and N(f_*|mu,s2) an approximation to the posterior
% marginal p(f_*|x_*,x,y) as returned by an inference method. The predictive
% distribution p(y_*|x_*,x,y) is approximated by.
%   q(y_*) = \int N(f_*|mu,s2) p(y_*|f_*) df_*
%
%   lp = log( q(y) ) for a particular value of y, if s2 is [] or 0, this
%                    corresponds to log( p(y|mu) )
%   ymu and ys2      the mean and variance of the predictive marginal q(y)
%                    note that these two numbers do not depend on a particular 
%                    value of y 
%  All vectors have the same size.
%
%
% 3) With five or six input arguments, the fifth being a string [INFERENCE MODE]
%
% [varargout] = lik(hyp, y, mu, s2, inf) OR
% [varargout] = lik(hyp, y, mu, s2, inf, i)
%
% There are three cases for inf, namely a) infLaplace, b) infEP and c) infVB. 
% The last input i, refers to derivatives w.r.t. the ith hyperparameter. 
%
% a1) [sum(lp),dlp,d2lp,d3lp] = lik(hyp, y, f, [], 'infLaplace')
% lp, dlp, d2lp and d3lp correspond to derivatives of the log likelihood 
% log(p(y|f)) w.r.t. to the latent location f.
%   lp = log( p(y|f) )
%  dlp = d   log( p(y|f) ) / df
% d2lp = d^2 log( p(y|f) ) / df^2
% d3lp = d^3 log( p(y|f) ) / df^3
%
% a2) [lp_dhyp,d2lp_dhyp] = lik(hyp, y, f, [], 'infLaplace', i)
% returns derivatives w.r.t. to the ith hyperparameter
%   lp_dhyp = d   log( p(y|f) ) / (df   dhyp_i)
% d2lp_dhyp = d^3 log( p(y|f) ) / (df^2 dhyp_i)
%
%
% b1) [lZ,dlZ,d2lZ] = lik(hyp, y, mu, s2, 'infEP')
% let Z = \int p(y|f) N(f|mu,s2) df then
%   lZ =     log(Z)
%  dlZ = d   log(Z) / dmu
% d2lZ = d^2 log(Z) / dmu^2
%
% b2) [dlZhyp] = lik(hyp, y, mu, s2, 'infEP', i)
% returns derivatives w.r.t. to the ith hyperparameter
% dlZhyp = d log(Z) / dhyp_i
%
%
% c1) [h,b,dh,db,d2h,d2b] = lik(hyp, y, [], ga, 'infVB')
% ga is the variance of a Gaussian lower bound to the likelihood p(y|f).
%   p(y|f) \ge exp( b*f - f.^2/(2*ga) - h(ga)/2 ) \propto N(f|b*ga,ga)
% The function returns the linear part b and the "scaling function" h(ga) and
% derivatives dh = d h/dga, db = d b/dga, d2h = d^2 h/dga and d2b = d^2 b/dga.
%
% c2) [dhhyp] = lik(hyp, y, [], ga, 'infVB', i)
% dhhyp = dh / dhyp_i, the derivative w.r.t. the ith hyperparameter
%
% Cumulative likelihoods are designed for binary classification. Therefore, they
% only look at the sign of the targets y; zero values are treated as +1.
%
% See the help for the individual likelihood for the computations specific to
% each likelihood function.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18
