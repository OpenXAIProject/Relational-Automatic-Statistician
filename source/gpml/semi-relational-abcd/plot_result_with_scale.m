function [fS_mean f_mean] = plot_result_with_scale(hyp, hyp_scale, hyp_fS, hyp_f,cov, cov_fS, cov_f, sn, x, y, z, model)
%PLOT_RESULT Summary of this function goes here
%   Detailed explanation goes here

N = model.N;
M = model.M;

noise_var = exp(2*sn);
for i = 1:M
    hyp_i = vertcat(hyp_scale{i}, hyp_fS, hyp_f{i});
    complete_sigma{i} = feval(cov{:}, hyp_i, x) + eye(N)*noise_var;
    complete_sigmastar{i} = feval(cov{:}, hyp_i, x, z);
    complete_sigmastarstar{i} = feval(cov{:}, hyp_i, z);
end


for i = 1:M
    complete_mean{i} = complete_sigmastar{i}' / complete_sigma{i} * y(:,i);
    complete_var{i} = diag(complete_sigmastarstar{i} - complete_sigmastar{i}' / complete_sigma{i} * complete_sigmastar{i});
    posterior_sigma{i} = complete_sigmastarstar{i} - complete_sigmastar{i}' / complete_sigma{i} * complete_sigmastar{i};
end

for i = 1:M
    scale_hyp_fS = vertcat(hyp_scale{i},hyp_fS);
    scale_cov_fS = {@covSum, {{@covConst}, {@covProd, {@covConst, cov_fS}}}};
    fS_sigma = feval(scale_cov_fS{:}, scale_hyp_fS, x);
    fS_sigma_star = feval(scale_cov_fS{:}, scale_hyp_fS, x, z);
    fS_sigma_starstar = feval(scale_cov_fS{:}, scale_hyp_fS, z);
    fS_mean{i} = fS_sigma_star' / complete_sigma{i}*y(:,i);
    fS_var{i} = diag(fS_sigma_starstar - fS_sigma_star'/complete_sigma{i}*fS_sigma_star);
end

for i = 1:M
    f_sigma = feval(cov_f{:}, hyp_f{i}, x);
    f_sigma_star = feval(cov_f{:}, hyp_f{i}, x, z);
    f_sigma_starstar = feval(cov_f{:}, hyp_f{i}, z);
    
    f_mean{i} = f_sigma_star' /complete_sigma{i}*y(:,i);
    f_var{i} = diag(f_sigma_starstar - f_sigma_star' / complete_sigma{i} * f_sigma_star);
end


for i = 1:M
    sub1 = subplot(M, 3, (i-1)*3 + 1);
    plot_a_figure(x,y(:,i), z, complete_mean{i}, complete_var{i}, sub1, ['Complete components for time series ' num2str(i)],1);
    sub2 = subplot(M, 3, (i-1)*3 + 2);
    plot_a_figure(x,y(:,i), z, fS_mean{i}, fS_var{i}, sub2, ['Shared component for time series ' num2str(i)],0);
    sub3 = subplot(M, 3, (i-1)*3 + 3);
    plot_a_figure(x,y(:,i), z, f_mean{i}, f_var{i}, sub3,  ['Distinctive component for time series ' num2str(i)],0);
end


end

function plot_a_figure(x, y, z, mu, sigma, sub, tit, draw_data)
lw = 2;
fontsize = 10;
opacity = 1;
fake_opacity = 0.1;
light_blue = [227 237 255]./255;
f = [mu+2*sqrt(sigma); 
flipdim(mu-2*sqrt(sigma),1)];
fill([z; flipdim(z,1)], f, [7 7 7]/8)
% jbfill(x', mu' + 2*sqrt(sigma)', mu' -2*sqrt(sigma)');
hold on; 
plot(z, mu, 'Color', colorbrew(2), 'LineWidth', lw); hold on;
if draw_data
    plot(x, y, 'k.'); hold on;
end
title(tit)
end

