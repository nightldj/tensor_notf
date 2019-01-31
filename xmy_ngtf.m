%% demo code for "Non-negative Factorization of the Occurrence Tensor from Financial Contracts", https://arxiv.org/pdf/1612.03350.pdf
% author: Zheng Xu
% contact: xuzhustc@gmail.com


function [cpX, outs] = xmy_ngtf(cpX0, tU2, opts)
tol = opts.tol;
maxiter2 = opts.maxiter2;

resv = zeros(maxiter2, 1); 
cpX = cpX0;
U2A = tenmat(tU2, 1); %matricazation
U2B = tenmat(tU2, 2);
U2C = tenmat(tU2, 3);
for j=1:maxiter2 %inner loop, decomposition
    cpXpre2 = cpX;
    Csq = cpX.U{3}'*cpX.U{3};
    tmpA = U2A.data*khatrirao(cpX.U{3}, cpX.U{2})/max(1e-20, Csq.*(cpX.U{2}'*cpX.U{2}));
    cpX.U{1} = max(tmpA, 0);
    Asq = cpX.U{1}'*cpX.U{1};
    tmpB = U2B.data*khatrirao(cpX.U{3}, cpX.U{1})/max(1e-20, (Csq.*Asq));
    cpX.U{2} = max(tmpB, 0);
    tmpC = U2C.data*khatrirao(cpX.U{2}, cpX.U{1})/max(1e-20, (cpX.U{2}'*cpX.U{2}).*Asq);
    cpX.U{3} = max(tmpC, 0);
    res2 = norm(cpX - cpXpre2)/max(1e-20, norm(cpXpre2));
    resv(j) = res2;
%     if opts.verbose
%         fprintf('NTF iter %d \t res %f\n', j, res2);
%     end
    if res2 < tol
        break;
    end
end %NTF loop
outs.res = resv(1:j);
end