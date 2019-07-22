function K = covPartialChangePointMultiD2(cov, hyp, x, z, i)
%COVPARTIALCHANGEPOINT Summary of this function goes here
%   Detailed explanation goes here

if ~numel(cov) == 2, error('Partial change point contains a dimension and a covariance'), end
dim = cov{1};
cov = cov{2};

for ii = 1:numel(cov)
    f = cov(ii); if iscell(f{:}), f = f{:}; end
    j(ii) = cellstr(feval(f{:}));
end

if nargin<3                                       
  K = ['2' '+' char(j(1))]; for ii=2:length(cov), K = [K, '+', char(j(ii))]; end, return
end

if nargin < 4, z = []; end;
original_z = z;
xeqz = numel(z) == 0;  dg = strcmp(z, 'diag') && numel(z) > 0;

if xeqz
    z = x;
end

v = [];               
for ii = 1:length(cov), v = [v repmat(ii, 1, eval(char(j(ii))))]; end

location = hyp(1);
steepness = exp(hyp(2));

tx = tanh((x(:,dim) - location)*steepness);
ax = 0.5 + 0.5*tx;

if ~dg
    ax = repmat(ax, 1, length(z(:,dim)));
end
if ~dg
    tz = tanh((z(:,dim)-location)*steepness);
    az = 0.5 + 0.5 * tz;
    az = repmat(az', length(x(:,dim)), 1);
else
    az = ax;
end

if nargin<5
    K = 0; if nargin==3, z = x; end
    for ii = 1:length(cov)
        f = cov(ii); if iscell(f{:}), f = f{:}; end
        if ii == 1
            K = K + ax .* feval(f{:}, hyp([false false (v==ii)]), x, original_z) .* az;
        end
    end
else
    if i==1
        dx = -0.5*(1-tx.^2)*steepness;
        dz = -0.5*(1-tz.^2)*steepness;
        dx = repmat(dx, 1, length(z(:,dim)));
        dz = repmat(dz', length(x(:,dim)), 1);
        dx = -dx;
        dz = -dz;
        K = 0;
        for ii = 1:length(cov)
            f = cov(ii); if iscell(f{:}), f = f{:}; end
            if ii == 1
                K = K + dx .* feval(f{:}, hyp([false false (v==ii)]), x, original_z) .* az;
                K = K + ax .* feval(f{:}, hyp([false false (v==ii)]), x, original_z) .* dz;
            end
        end
    elseif i==2
        dx = +0.5*(1-tx.^2).*(x(:,dim)-location).*steepness;
        dz = +0.5*(1-tz.^2).*(z(:,dim)-location).*steepness;
        dx = repmat(dx, 1, length(z(:,dim)));
        dz = repmat(dz', length(x(:,dim)), 1);
        dx = -dx; 
        dz = -dz;
        K = 0;
        for ii = 1:length(cov)                              
            f = cov(ii); if iscell(f{:}), f = f{:}; end 
            if ii == 1
                K = K + dx .* feval(f{:}, hyp([false false (v==ii)]), x, original_z) .* az;
                K = K + ax .* feval(f{:}, hyp([false false (v==ii)]), x, original_z) .* dz;
            end
        end
    elseif i < length(v) +3
        i = i - 2;
        vi = v(i);                                       
        j = sum(v(1:i)==vi);                    
        f  = cov(vi);
        if iscell(f{:}), f = f{:}; end         
        K = feval(f{:}, hyp([false false (v==vi)]), x, original_z, j);                  
        if vi == 1
            K = ax .* K .* az;
        elseif vi ==2
            K = (1-ax) .* K .* (1-az);
        end
    else
        error('Unknown hyperparameter')
    end
end


end

