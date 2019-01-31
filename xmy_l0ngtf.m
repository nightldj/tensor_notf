%% demo code for "Non-negative Factorization of the Occurrence Tensor from Financial Contracts", https://arxiv.org/pdf/1612.03350.pdf
% author: Zheng Xu
% contact: xuzhustc@gmail.com


function [cpX, outs] = xmy_l0ngtf(normprox, tO, cpX0, lam0, opts)
maxiter = opts.maxiter;
%maxiter2 = opts.maxiter2;
tol = opts.tol;
tau = opts.tau;

resv = zeros(maxiter, 1); 
presv = zeros(maxiter, 1); 
dresv = zeros(maxiter, 1); 
objv = zeros(maxiter, 1); 
cpX = cpX0;
lam = lam0;
for i=1:maxiter %outer loop, ADMM
    cpXpre1 = cpX;
    lampre = lam;
    tU = normprox(tensor(cpX) - tO - lam, 1.0/tau);
%     if opts.verbose
%         fprintf('tU sparseness: %f\n', sum(reshape(abs(tU.data), [], 1) > 1e-20 )/numel(tU.data));
%     end
    tU2 = tU+tO+lam;
    [cpX, ~] = xmy_ngtf(cpX, tU2, opts);
    lam = lam + tU + tO - tensor(cpX);
    res1 = norm(cpX - cpXpre1)/max(1e-20, norm(cpXpre1));
    res2 = norm(lam-lampre)/max(1e-20, norm(lampre));
    res = max(res1, res2);
    resv(i) = res;
    presv(i) = res2;
    dresv(i) = res1;
    if opts.verbose
        objv(i) = opts.obj(cpX);
        fprintf('NGTF iter %d \t rel res %f \t l0 error %f\n', i, res, objv(i));
    end
    if res < tol
        break;
    end
end
outs.res = resv(1:i);
outs.pres = presv(1:i);
outs.dres = dresv(1:i);
outs.objs = objv(1:i);
outs.iter = i;
end