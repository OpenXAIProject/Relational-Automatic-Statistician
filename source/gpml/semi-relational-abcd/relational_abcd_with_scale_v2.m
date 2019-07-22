%Copyright 2018 UNIST under XAI Project supported by Ministry of Science and ICT, Korea

%Licensed under the Apache License, Version 2.0 (the "License"); 
%you may not use this file except in compliance with the License.
%You may obtain a copy of the License at

%   https://www.apache.org/licenses/LICENSE-2.0

%Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

function [ varargout ] = relational_abcd_with_scale_v2(hyp, model, sn,  cov_fS, cov_f, x, y, xs, ys)
%RELATIONAL_ABCD Summary of this function goes here
%   Detailed explanation goes here

if nargin < 7 || nargin > 9
    disp('Usage [nlZ ndlZ] = relationa_abcd(hyp_fS, hyp_f, x, y, xs, ys)');
end



try
if nargin > 7
    %do something
else
    if nargout == 1
        nlZ_S = inf_scale(hyp, mean, cov, lik, x, y); 
        varargout = nlZ_S;
        return;
    else
        [nlZ, dnlZ, nlZ_S] = inf_scale(hyp, sn, cov_fS, cov_f, x, y, model);
    end
end
catch
    msgstr = lasterr;
    warning('Inference method failed [%s] .. attempting to continue',msgstr)
    varargout = {9223372036854775807, 0*hyp, 9223372036854775807}; return % return big number in python
end

if nargin == 7
    varargout = {nlZ, dnlZ, nlZ_S};
end
    


end

function [nlZ dnlZ nlZ_S] = inf_scale(hyp0, sn, cov_fS, cov_f, x, y, model)

[scale, hyp_fS, hyp_f] = seperate_hyp_with_scale(model, hyp0);

[N, D] = size(x);
M = size(y,2);
sn2 = exp(2*sn);


cov_fSi = {@covSum, {{@covConst}, {@covProd, {@covConst, cov_fS}}}};
if nargout == 3
    nlZ_S = 0;
    for i = 1:M
        hyp_fSi = vertcat(scale{i}, hyp_fS);
        K_S = feval(cov_fSi{:}, hyp_fSi, x);
        L_S = chol(K_S/sn2+eye(N));
        alpha_S = solve_chol(L_S,y(:,i))/sn2;
        nlZ_S = nlZ_S + y(:,i)'*alpha_S/2 + sum(log(diag(L_S))) + N*log(2*pi*sn2)/2;
    end
end
% cov = {@covSum, {{@covConst}, {@covProd, {@covConst, cov_fS}}, cov_f}};
cov = {@covSum, {{@covSum, {{@covConst}, {@covProd, {@covConst, cov_fS}}}}, cov_f}};

for i = 1:M
    hyp{i} = vertcat(scale{i}, hyp_fS, hyp_f{i}); % calculation here
end

for i = 1:M
    K{i} = feval(cov{:}, hyp{i}, x);
    L{i} = chol(K{i}/sn2+eye(N)); 
    alpha{i} = solve_chol(L{i},y(:,i))/sn2;
end

if nargout > 1
    nlZ = 0;
    for m = 1:M
        nlZ = nlZ + y(:,m)'*alpha{m}/2 + sum(log(diag(L{m}))) + N*log(2*pi*sn2)/2;
    end
    if nargout > 2
        dnlZ = zeros(size(hyp0)); % input here
        for m = 1:M
            Q{m} = solve_chol(L{m},eye(N))/sn2 - alpha{m}*alpha{m}';
        end
        
        for m = 1:M
            dnlZ(2*m - 1) = sum(sum(Q{m}.*feval(cov{:}, hyp{m}, x, [], 1)))/2;
            dnlZ(2*m) = sum(sum(Q{m}.*feval(cov{:}, hyp{m}, x, [], 2)))/2;
        end
        for i = 1: numel(hyp_fS)
            for m = 1:M
                dnlZ(2*M + i) = dnlZ(2*M + i) + sum(sum(Q{m}.*feval(cov{:}, hyp{m}, x, [], 2 + i)))/2;
            end
        end
        index = 2*M + numel(hyp_fS);
        for m = 1:M
            for i = 1:length(hyp_f{m})
                index = index + 1;
                dnlZ(index) = sum(sum(Q{m}.*feval(cov{:}, hyp{m}, x, [], 2 + numel(hyp_fS) + i)))/2;
            end
        end
    end
end

end

