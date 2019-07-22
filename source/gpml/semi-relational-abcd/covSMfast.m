function K = covSMfast(Q, hyp, x, z, i)

% Gaussian Spectral Mixture covariance function. The 
% covariance function is parameterized as:
%
% k(x^p,x^q) = w'*prod( exp(-2*pi^2*d^2*v)*cos(2*pi*d*m), 2), d = |x^p,x^q|
%
% where m(DxQ), v(DxQ) are the means and variances of the spectral mixture
% components and w are the mixture weights. The hyperparameters are:
%
% hyp = [ log(w)
%         log(m(:))
%         log(sqrt(v(:)))  ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% Copyright (c) by Andrew Gordon Wilson and Hannes Nickisch, 2013-10-09.
%
% For more details, see 
% 1) Gaussian Process Kernels for Pattern Discovery and Extrapolation,
% ICML, 2013, by Andrew Gordon Wilson and Ryan Prescott Adams.
% 2) GPatt: Fast Multidimensional Pattern Extrapolation with Gaussian 
% Processes, arXiv 1310.5288, 2013, by Andrew Gordon Wilson, Elad Gilboa, 
% Arye Nehorai and John P. Cunningham, and
% http://mlg.eng.cam.ac.uk/andrew/pattern
%
% See also COVFUNCTIONS.M.




if nargin<3, K = sprintf('%d + 2*D*%d',Q,Q); return; end   % report no of params
if nargin<4, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

[n,D] = size(x); Q = length(hyp)/(1+2*D);
w = exp(hyp(1:Q));                                             % mixture weights
m = exp(reshape(hyp(Q+(1:Q*D)),D,Q));                          % mixture centers
v = exp(reshape(2*hyp(Q+Q*D+(1:Q*D)),D,Q));                  % mixture variances

if dg                                              % compute squared distance d2
  d2 = zeros([n,1,D]);
else
  if xeqz                                                 % symmetric matrix Kxx
    d2 = zeros([n,n,D]);
    for j=1:D, d2(:,:,j) = sq_dist(x(:,j)'); end
  else                                                   % cross covariances Kxz
    d2 = zeros([n,size(z,1),D]);
    for j=1:D, d2(:,:,j) = sq_dist(x(:,j)',z(:,j)'); end
  end
end
d = sqrt(d2);                                         % compute plain distance d

k  = @(d2v,dm) exp(-2*pi^2*d2v).*cos(2*pi*dm);    % evaluation of the covariance
km = @(dm) -2*pi*tan(2*pi*dm).*dm;     % remainder when differentiating w.r.t. m
kv = @(d2v) -(2*pi)^2*d2v;             % remainder when differentiating w.r.t. v

K = 0;
if nargin<5                                       % evaluation of the covariance
  c = 1;                                                   % initial value for C
  qq = 1:Q;                                          % indices q to iterate over
elseif i<=Q                                               % derivatives w.r.t. w
  c = 1;
  qq = i;
elseif i<=Q*D+Q                                           % derivatives w.r.t. m
  p = mod(i-Q-1,D)+1; q = (i-Q-p)/D+1; c = km(d(:,:,p)*m(p,q));
  qq = q;
elseif i<=2*Q*D+Q                                         % derivatives w.r.t. v
  p = mod(i-(D+1)*Q-1,D)+1; q = (i-(D+1)*Q-p)/D+1; c = kv(d2(:,:,p)*v(p,q));
  qq = q;
end
for q=qq
  C = w(q)*c;
  for j=1:D, C = C.*k(d2(:,:,j)*v(j,q),d(:,:,j)*m(j,q)); end
  K = K+C;
end