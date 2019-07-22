%Copyright 2018 UNIST under XAI Project supported by Ministry of Science and ICT, Korea

%Licensed under the Apache License, Version 2.0 (the "License"); 
%you may not use this file except in compliance with the License.
%You may obtain a copy of the License at

%   https://www.apache.org/licenses/LICENSE-2.0

%Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


function [ varargout ] = relational_abcd(model, hyp, sn,  cov_fS, cov_f, x, y, xs, ys)
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
        [post nlZ] = inf(hyp, mean, cov, lik, x, y); dnlZ = {};
    else
        [post nlZ dnlZ] = inf(hyp, sn, cov_fS, cov_f, x, y, model);
    end
end
catch
    msgstr = lasterr;
    warning('Inference method failed [%s] .. attempting to continue',msgstr)
    varargout = {NaN, 0*hyp, 0}; return
end

if nargin == 7
    varargout = {nlZ, dnlZ, post};
end
    


end

function [post nlZ dnlZ] = inf(hyp0, sn, cov_fS, cov_f, x, y, model)

[hyp_fS, hyp_f] = seperate_hyp(model, hyp0);
[N, D] = size(x);
M = size(y,2);
cov = {@covSum, {cov_fS, cov_f}};
for i = 1:M
    hyp{i} = vertcat(hyp_fS, hyp_f{i}); % calculation here
end
sn2 = exp(2*sn);
for i = 1:M
    K{i} = feval(cov{:}, hyp{i}, x);
    L{i} = chol(K{i}/sn2+eye(N)); 
    alpha{i} = solve_chol(L{i},y(:,i))/sn2;
end

post.alpha = alpha;
post.L = L;

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
        for i = 1: numel(hyp_fS)
            dnlZ(i) = 0;
            for m = 1:M
                dnlZ(i) = dnlZ(i) + sum(sum(Q{m}.*feval(cov{:}, hyp{m}, x, [], i)))/2;
            end
        end
        index = numel(hyp_fS);
        for m = 1:M
            for i = 1:length(hyp_f{m})
                index = index + 1;
                dnlZ(index) = sum(sum(Q{m}.*feval(cov{:}, hyp{m}, x, [], numel(hyp_fS)+ i)))/2;
            end
        end
    end
end

end

