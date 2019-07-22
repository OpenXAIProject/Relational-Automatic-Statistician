clear all;

% cov_fS = {@covChangePointMultiD, {1, {@covLinear}, {@covConst}}};
% hyp_fS = [0.5 0.5 -.9 -1 1 0.1];

cov_fS = {@covNoise};
hyp_fS = [-0.1];


cov_f1 = {@covSEiso};
hyp_f1 = [-0.8 - 0.8];
cov_f2 = {@covSEiso};
hyp_f2 = [-.1 -.1];

% cov1 = {@covSum, {cov_fS, cov_f1}};
% cov2 = {@covSum, {cov_fS, cov_f2}};
% 
% hyp1 = [hyp_fS  hyp_f1];
% hyp2 = [hyp_fS  hyp_f2];
n=100;
x = linspace(1,10,100);


cov1 = {@covLinear};
cov2 = {@covLinear};
hyp1 = [0.1 8];
hyp2 = [0.1 -8];

K1 = feval(cov1{:}, hyp1, x);
K2 = feval(cov2{:}, hyp2, x);

t = gpml_randn(0.15, n, 1);
y1 = chol(K1)'* t;
y2 = chol(K2)'* t;

plot(x,y1);hold on; plot(x, y2);

y = [y1 y2];

x = x';

covfunc_fS = {@covNoise};
hyp_fS = [0.5];

covfunc_f = {@covSMfast ,10};
% covfunc_f = {@covSEiso};

% hyp = [-0.1 -0.1 -0.1 -0.1];

hyp = [];
M = size(y,2);
for i = 1:M
    hyp = [hyp hyp_SM_init_1D('covSMfast' ,10,x ,y(:,i))'];
end


model.N = size(x,1);
model.M = size(y,2);
model.D = size(x,2);
model.num_hyp_fS = numel(hyp_fS);
model.num_hyp_f = numel(hyp);
hyp = [hyp_fS hyp];
sn = log(0.1);

X = x;
Y = y;

opts = struct('Display', 'off', 'Method', 'lbfgs', 'MaxIter', 200, ...
    'MaxFunEvals', 200, 'DerivativeCheck', 'off');
func = @(x) relational_abcd(model, x, sn, covfunc_fS, covfunc_f, X,Y);
hyp = minFunc(func, hyp', opts);

[hyp_fS, hyp_f] = seperate_hyp(model, hyp);

cov = {@covSum, {covfunc_fS, covfunc_f}};

x2 = linspace(10,11,10); 
z = vertcat(x, x2');
[fS_mean f_mean] = plot_result(hyp, hyp_fS, hyp_f, cov, covfunc_fS, covfunc_f, sn, x, y, z, model);
for i = 1:model.M
    y_hat{i} = fS_mean{i} + f_mean{i};
end
