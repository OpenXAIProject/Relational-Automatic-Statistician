clear all;

cov_fS = {@covPartialChangePointMultiD, {1, {@covConst}}};
hyp_fS = [5 0.2 2];

cov_f1 = {@covLinear};
cov_f2 = {@covLinear};

hyp_f1 = [-2 2];
hyp_f2 = [2 2];


n = 100;
x = linspace(1,10,100)';

sn = 0.1;

cov1 = {@covSum, {cov_fS, cov_f1}};
cov2 = {@covSum, {cov_fS, cov_f2}};

hyp1 = [hyp_fS hyp_f1];
hyp2 = [hyp_fS hyp_f2];


K1 = feval(cov1{:}, hyp1, x) + sn * eye(n);
K2 = feval(cov2{:}, hyp2, x) + sn * eye(n);
% t1 = gpml_randn(0.15,n,1);
y1 = chol(K1)'*gpml_randn(0.15,n,1);
y2 = chol(K2)'*gpml_randn(0.15,n,1);

plot(x,y1);hold on; plot(x,y2);

y = [y1 y2];

covfunc_f = {@covSMfast,10};

hyp = [];
M = size(y,2);

for i = 1:M
    hyp = [hyp hyp_SM_init_1D('covSMfast', 10, x, zeros(size(y(:,i))))'];
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
func = @(x) relational_abcd(model, x, sn, cov_fS, covfunc_f, X,Y);
hyp = minFunc(func, hyp', opts);

[hyp_fS, hyp_f] = seperate_hyp(model, hyp);

cov = {@covSum, {cov_fS, covfunc_f}};

x2 = linspace(10,11,10); 
z = vertcat(x, x2');
[fS_mean f_mean] = plot_result(hyp, hyp_fS, hyp_f, cov, cov_fS, covfunc_f, sn, x, y, z, model);
for i = 1:model.M
    y_hat{i} = fS_mean{i} + f_mean{i};
end