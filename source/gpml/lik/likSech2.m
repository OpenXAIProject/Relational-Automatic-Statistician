function [varargout] = likSech2(hyp, y, mu, s2, inf, i)

% likSech2 - sech-square likelihood function for regression. Often, the sech-
% square distribution is also referred to as the logistic distribution not to be
% confused with the logistic function for classification. The expression for the
% likelihood is 
%   likSech2(t) = Z / cosh(tau*(y-t))^2 where 
% tau = pi/(2*sqrt(3)*sn) and Z = tau/2
% and y is the mean and sn^2 is the variance.
%
% hyp = [ log(sn)  ]
%
% Several modes are provided, for computing likelihoods, derivatives and moments
% respectively, see likelihoods.m for the details. In general, care is taken
% to avoid numerical issues when the arguments are extreme. The moments
% \int f^k likSech2(y,f) N(f|mu,var) df are calculated via a Gaussian
% scale mixture approximation.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-07-21.
%
% See also likFunctions.m, likLogistic.m.

if nargin<2, varargout = {'1'}; return; end   % report number of hyperparameters

sn = exp(hyp); tau = pi/(2*sqrt(3)*sn);
lZ = log(pi) - log(sn) - log(4*sqrt(3));

if nargin<5                              % prediction mode if inf is not present
  if numel(y)==0,  y = zeros(size(mu)); end
  s2zero = 1; if nargin>3, if norm(s2)>0, s2zero = 0; end, end         % s2==0 ?
  if s2zero                                         % log probability evaluation
    lp = lZ - 2*logcosh(tau*(y-mu)); s2 = 0;
  else                                                              % prediction
    lp = likSech2(hyp, y, mu, s2, 'infEP');
  end
  ymu = {}; ys2 = {};
  if nargout>1
    ymu = mu;                                                   % first y moment
    if nargout>2
      ys2 = s2 + sn.^2;                                        % second y moment
    end
  end
  varargout = {lp,ymu,ys2};
else                                                            % inference mode
  switch inf 
  case 'infLaplace'
    r = y-mu; [g,dg,d2g,d3g] = logcosh(tau.*r);         % precompute derivatives
    if nargin<6                                             % no derivative mode
      dlp = {}; d2lp = {}; d3lp = {};
      lp = lZ - 2*g;
      if nargout>1                                           % first derivatives
        dlp = 2*tau.*dg;
        if nargout>2                          % 2nd derivative of log likelihood
          d2lp = -2*tau.^2.*d2g;
          if nargout>3                        % 3rd derivative of log likelihood
            d3lp = 2*tau.^3.*d3g;
          end
        end
      end
      varargout = {sum(lp),dlp,d2lp,d3lp};
    else                                                       % derivative mode
      lp_dhyp = 2*tau.*r.*dg - 1;                         % derivative w.r.t. sn
      d2lp_dhyp = 2*tau.^2.*(2*d2g + tau.*r.*d3g);
      varargout = {lp_dhyp,d2lp_dhyp};
    end
    
  case 'infEP'
    n = max([length(y),length(mu),length(s2),length(sn)]); on = ones(n,1);
    y = y.*on; mu = mu.*on; s2 = s2.*on; sn = sn.*on;             % vectors only
    fac = 1e3;          % factor between the widths of the two distributions ...
       % ... from when one considered a delta peak, we use 3 orders of magnitude
    idlik = fac*sn<sqrt(s2);                        % Likelihood is a delta peak
    idgau = fac*sqrt(s2)<sn;                          % Gaussian is a delta peak
    id = ~idgau & ~idlik;                          % interesting case in between
    % likLogistic(t)   \approx 1/2 + \sum_{i=1}^5 (c_i/2) erf(lam_i/sqrt(2)t)
    % likSech2(t|y,sn) \approx \sum_{i=1}^5 c_i likGauss(t|y,sn*rho_i)
    lam = sqrt(2)*[0.44 0.41 0.40 0.39 0.36];  % approx coeffs lam_i, c_i, rho_i
    c   = [1.146480988574439e+02; -1.508871030070582e+03; 2.676085036831241e+03;
          -1.356294962039222e+03;  7.543285642111850e+01                      ];
    rho = sqrt(3)./(pi*lam); o5 = ones(1,5);
    if nargin<6                                             % no derivative mode
      lZ = zeros(n,1); dlZ = lZ; d2lZ = lZ;                    % allocate memory
      if any(idlik)
        [lZ(idlik),dlZ(idlik),d2lZ(idlik)] = ...
                                likGauss(log(s2(idlik))/2, mu(idlik), y(idlik));
      end
      if any(idgau)
        [lZ(idgau),dlZ(idgau),d2lZ(idgau)] = ...
                               likSech2(log(sn(idgau)), mu(idgau), y(idgau));
      end
      if any(id)
        [lZc,dlZc,d2lZc] = likGauss(log(sn(id)*rho), ...
                                           y(id)*o5, mu(id)*o5, s2(id)*o5, inf);
        lZ(id) = log_expA_x(lZc,c);            % A=lZc, B=dlZc, lZ=log(exp(A)*c)
        dlZ(id)  = expABz_expAx(lZc, c, dlZc, c);  % ((exp(A).*B)*c)./(exp(A)*c)
        % d2lZ(id) = ((exp(A).*Z)*c)./(exp(A)*c) - dlZ.^2
        d2lZ(id) = expABz_expAx(lZc, c, dlZc.^2+d2lZc, c) - dlZ(id).^2;
        
        % the tail asymptotics of likSech2 is the same as for likLaplace
        % which is not covered by the scale mixture approximation, so for
        % extreme values, we approximate likSech2 by a rescaled likLaplace
        tmu = (mu-y)./sn; tvar = s2./sn.^2; crit = 0.596*(abs(tmu)-5.38)-tvar;
        idl = -1<crit & id;                       % if 0<crit, Laplace is better
        if any(idl)                         % close to zero, we use a smooth ..
          lam = 1./(1+exp(-15*crit(idl)));   % .. interpolation with weights lam
          thyp = log(sqrt(6)*sn(idl)/pi);
          [lZl,dlZl,d2lZl] = likLaplace(thyp, y(idl), mu(idl), s2(idl), inf);
          lZ(idl)   = (1-lam).*lZ(idl)   + lam.*lZl;
          dlZ(idl)  = (1-lam).*dlZ(idl)  + lam.*dlZl;
          d2lZ(idl) = (1-lam).*d2lZ(idl) + lam.*d2lZl;
        end
      end      
      varargout = {lZ,dlZ,d2lZ};
    else                                                       % derivative mode
      dlZhyp = zeros(n,1);
      if any(idlik)
        dlZhyp(idlik) = 0;
      end
      if any(idgau)
        dlZhyp(idgau) = ...
              likSech2(log(sn(idgau)), mu(idgau), y(idgau), 'infLaplace', 1);
      end
      if any(id)
        lZc     = likGauss(log(sn(id)*rho),y(id)*o5,mu(id)*o5,s2(id)*o5,inf);
        dlZhypc = likGauss(log(sn(id)*rho),y(id)*o5,mu(id)*o5,s2(id)*o5,inf,1);
        % dlZhyp = ((exp(lZc).*dlZhypc)*c)./(exp(lZc)*c)
        dlZhyp(id) = expABz_expAx(lZc, c, dlZhypc, c);
        
        % the tail asymptotics of likSech2 is the same as for likLaplace
        % which is not covered by the scale mixture approximation, so for
        % extreme values, we approximate likLogistic by a rescaled likLaplace
        tmu = (mu-y)./sn; tvar = s2./sn.^2; crit = 0.596*(abs(tmu)-5.38)-tvar;
        idl = -1<crit & id;                       % if 0<crit, Laplace is better
        if any(idl)                         % close to zero, we use a smooth ..
          lam = 1./(1+exp(-15*crit(idl)));   % .. interpolation with weights lam
          thyp = log(sqrt(6)*sn(idl)/pi);
          dlZhypl = likLaplace(thyp, y(idl), mu(idl), s2(idl), inf, i);
          dlZhyp(idl) = (1-lam).*dlZhyp(idl) + lam.*dlZhypl;
        end
      end
      varargout = {dlZhyp};                           % derivative w.r.t. hypers
    end
    
  case 'infVB'
    if nargin<6
      % variational lower site bound
      % using -log( 2*cosh(s/2) );
      % the bound has the form: b*s - s.^2/(2*ga) - h(ga)/2 with b=1/2
      ga = s2; n = numel(ga); b = y./ga; y = y.*ones(n,1);
      db = -y./ga.^2; d2b = 2*y./ga.^3;
      h = zeros(n,1); dh = h; d2h = h;     % allocate memory for return argument
      idup = 50<=ga(:);                   % asymptotic behavior for large gammas
      h(idup) = 4*tau^2*ga(idup) -4*log(2);
      id = ga(:)>1/(2*tau^2) & ~idup;           % interesting zone is in between
      [g,dg,d2g] = inv_xcothx(2*tau^2*ga(id));
      thg = tanh(g);
      h(id) = 4*logcosh(g) -2*g.*thg;
      h = h + y.^2./ga - 2*lZ;
      g1mt2 = g.*(1-thg.^2);                                 % first derivatives
      dh(idup) = 4*tau^2;
      dh(id) = 4*tau^2*dg.*( thg - g1mt2 );
      dh = dh - (y./ga).^2;
      d2h(id) = 8*tau^4*( g1mt2.*(2*dg.^2.*thg-d2g) +d2g.*thg ); % second derivs
      d2h = d2h + 2*y.^2./ga.^3;
      id = ga<0; h(id) = Inf; dh(id) = 0; d2h(id) = 0;     % neg. var. treatment
      varargout = {h,b,dh,db,d2h,d2b};
    else
      ga = s2; n = numel(ga); 
      dhhyp = zeros(n,1);
      idup = 50<=ga(:);                   % asymptotic behavior for large gammas
      dhhyp(idup) = -2*pi^2/3/sn^2*ga(idup);           % tau = pi/(2*sqrt(3)*sn)
      id = ga(:)>1/(2*tau^2) & ~idup;           % interesting zone is in between
      [g,dg] = inv_xcothx(2*tau^2*ga(id)); thg = tanh(g); [f,df] = logcosh(g);
      % h = 4*f -2*g.*tanh(g)
      if any(id)
        dhhyp(id) = 8*tau^2*ga(id).*dg.*( thg - 2*df + g.*(1-thg.^2) ); 
      end
      dhhyp = dhhyp + 2; %          from lZ = log(pi) - log(sn) - log(4*sqrt(3))
      dhhyp(ga<0) = 0;              % negative variances get a special treatment
      varargout = {dhhyp};                                  % deriv. wrt hyp.lik
    end
  end
end

% Invert f(x) = x.*coth(x) = y, return the positive value
% uses Newton's method to minimize (y-f(x))² w.r.t. x
function [x,dx,d2x] = inv_xcothx(y)
%   f = @(x) x.*coth(x);
%  df = @(x) x + (1-f(x)).*coth(x);
% d2f = @(x) 2*(1-f(x)).*(1-coth(x).^2);
if numel(y)==0, x=[]; dx=[]; d2x=[]; return, end
x  = sqrt(y.^2-1); i = 1; % init
ep = eps./(1+abs(x)); th=tanh(x); fx=(x+ep)./(th+ep);                 % function
r  = fx-y; % init
while i==1 || (i<10 && max(abs(r))>1e-12)
    dfx  = x + (1-fx)./(th+ep);                               % first derivative
    d2fx = 2*(  1-fx + (x-th+ep/3)./(th.*th.*th+ep) );       % second derivative
    x = x - r.*dfx./(dfx.^2+r.*d2fx+ep);                           % Newton step
    ep = eps./(1+abs(x)); th=tanh(x); fx=(x+ep)./(th+ep);
    r  = fx-y; i = i+1;
end
% first derivative dx/dy; derivatives of inverse functions are reciprocals
dx = 1./dfx;
% second derivative d2x/dy2; quotient rule and chaine rule
d2x = -d2fx.*dx./dfx./dfx;

% numerically safe version of log(cosh(x)) = log(exp(x)+exp(-x))-log(2)
function [f,df,d2f,d3f] = logcosh(x)
   a  = exp(-2*abs(x));  % always between 0 and 1 and therefore safe to evaluate
   f  = abs(x)  + log(1+a) - log(2);
  df  = sign(x).*( 1 - 2*a./(1+a) );
  d2f = 4*a./(1+a).^2;
  d3f = -8*sign(x).*a.*(1-a)./(1+a).^3;

%  computes y = log( exp(A)*x ) in a numerically safe way by subtracting the
%  maximal value in each row to avoid cancelation after taking the exp
function y = log_expA_x(A,x)
  N = size(A,2);  maxA = max(A,[],2);      % number of columns, max over columns
  y = log(exp(A-maxA*ones(1,N))*x) + maxA;  % exp(A) = exp(A-max(A))*exp(max(A))
  
%  computes y = ( (exp(A).*B)*z ) ./ ( exp(A)*x ) in a numerically safe way
%  The function is not general in the sense that it yields correct values for
%  all types of inputs. We assume that the values are close together.
function y = expABz_expAx(A,x,B,z)
  N = size(A,2);  maxA = max(A,[],2);      % number of columns, max over columns
  A = A-maxA*ones(1,N);                                 % subtract maximum value
  y = ( (exp(A).*B)*z ) ./ ( exp(A)*x );