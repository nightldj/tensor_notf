%% demo code for "Non-negative Factorization of the Occurrence Tensor from Financial Contracts", https://arxiv.org/pdf/1612.03350.pdf
% author: Zheng Xu
% contact: xuzhustc@gmail.com


close all;
clear;
clc;
addpath('./tensor_toolbox');
addpath('./tensor_toolbox/met');


%% synthetic problem
rng(2016)
R = 3; %3; % rank, three communities
NA = 50; %50; %dim
sa = 0.7; %0.7; %sparseness, higher means sparser
NB = 20; %20;
sb = 0.5; %0.5;
NC = 10; %10;
sc = 0.3; %0.3
A = max(rand(NA, R)-sa, 0);
B = max(rand(NB, R)-sb, 0);
C = max(rand(NC, R)-sc, 0);
A = A./repmat(max(sum(A), 1e-20), size(A,1), 1);
B = B./repmat(max(sum(B), 1e-20), size(B,1), 1);
C = C./repmat(max(sum(C), 1e-20), size(C,1), 1);
cpX = ktensor({A,B,C});
dtX = double(cpX) >0 ;
%tX = tensor(cpX);
fprintf('sparseness, A: %.4f \t B: %.4f \t C: %.4f \n', 1-sum(A(:)>0)/numel(A), 1-sum(B(:)>0)/numel(B), 1-sum(C(:)>0)/numel(C));
fprintf('sparseness, low rank tensor: %.4f \n', 1-sum(dtX(:))/numel(dtX));


%nrts = 0: .005: .1; %vary noise ration
nrts = 3; % this is actually for CP rank parameter
nrt = .1;
conits = zeros(size(nrts));
ofps = zeros(size(conits));
ofns = zeros(size(conits));
xfps = zeros(size(conits));
xfns = zeros(size(conits));
omses = zeros(size(conits));
xmses = zeros(size(conits));
for i = 1:length(nrts)
    R = nrts(i);
    
    dtO = dtX;
    noise = (rand(size(dtO)) - 1 + nrt) >0; %0.01 noise
    fprintf('sparseness, noise: %.4f \n', 1-sum(noise(:))/numel(noise));
    dtO = abs(dtO - noise);
    fprintf('sparseness, observe tensor: %.4f \n', 1-sum(dtO(:))/numel(dtO));
    tO = tensor(dtO);
    
    %% parameter
    opts.maxiter = 1000;
    opts.maxiter2 = 10;
    opts.tol = 1e-3;
    opts.tau = 10;
    opts.verbose = 0;
    
    
    %% norm proximal
    proxl0 = @(z,t) ( (z.^2-2*t) > 1e-20 ).*z;
    normprox = proxl0; %proxl0;
    
    opts.obj = @(cpX) sum(reshape(abs((double(tensor(cpX))> 1e-20) - dtX), [], 1) > 1e-20 );
    
    %% initialization
    cpX0 = cp_als(tO, R, 'printitn', 0);
    cpX0.U{3} = cpX0.U{3}.*repmat(cpX0.lambda', size(cpX0.U{3}, 1), 1);
    cpX0.lambda = ones(R, 1);
    lam0 = tenzeros(size(tO));
    
    %% optimize
    [cpX, outs] = xmy_l0ngtf(normprox, tO, cpX0, lam0, opts);
    
    fprintf('observation error: %d\n', sum(reshape(abs(tO.data - dtX), [], 1) > 1e-20 ));
    mseX = norm(tensor(dtX)-tensor(cpX))^2/numel(dtX);
    mseO = norm(tO-tensor(cpX))^2/numel(tO.data);
    fprintf('reconstruct mse: O:%f \t X:%f \n', mseO, mseX);
    errcntO = abs((double(tensor(cpX))> 1e-20) - tO.data) > 1e-20;
    errcntX = abs((double(tensor(cpX))> 1e-20) - dtX) > 1e-20;
    fprintf('reconstruct error count: O:%f \t X:%f \n', sum(errcntO(:)), sum(errcntX(:)));
    fprintf('sparseness, A: %.4f \t B: %.4f \t C: %.4f \n', 1-sum(cpX.U{1}(:)>1e-20)/numel(cpX.U{1}), 1-sum(cpX.U{2}(:)>1e-20)/numel(cpX.U{2}), 1-sum(cpX.U{3}(:)>1e-20)/numel(cpX.U{3}));
    
    conits(i) = outs.iter;
    ofps(i) = sum(errcntO(tO.data < .5));
    ofns(i) = sum(errcntO(tO.data > .5));
    xfps(i) = sum(errcntX(dtX < .5));
    xfns(i) = sum(errcntX(dtX > .5));
    omses(i) = mseO;
    xmses(i) = mseX;
    
end
