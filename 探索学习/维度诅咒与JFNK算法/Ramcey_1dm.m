a=[1 2;3 4]

a(:)

clear;
clc;
close all;

p=struct();
p.beta=0.95;
p.alpha=0.35;
p.delta=0.08;
p.tol=1e-6;
p.maxiter=10000;
p

capitalimin=0.1;
capitalmax=5;
capitalgridcount=200;
capitalgrid=linspace(capitalimin,capitalmax,capitalgridcount)';

productivitystate=[0.9;
                    1];
productivitystatecount=length(productivitystate);
transitionmatrix=[0.9 0.1;0.1 0.9];

初始化值函数和策略函数
valuefunction=zeros(capitalgridcount,productivitystatecount);
updatefunction=zeros(capitalgridcount,productivitystatecount);
policycapital = zeros(capitalgridcount, productivitystatecount);
policyindex = ones(capitalgridcount, productivitystatecount);

做一次更新
% for productivityindex=1:productivitystatecount
%     currentproductivity=productivitystate(productivityindex);
%     expected=valuefunction*transitionmatrix(productivityindex,:)';
%     for capitalindex=1:capitalgridcount
%         currentcapital=capitalgrid(capitalindex);
%          y=currentproductivity*currentcapital^p.alpha;
%          c=y+(1-p.delta)*currentcapital-capitalgrid;
%          currentU=-inf(capitalgridcount,1);
%          feasible=c>0;
%          currentU(feasible)=log(c(feasible));%这是一种常用的逻辑索引，学习一下
%          Candidatevalue=currentU+p.beta*expected;
%          [bestvalue,bestindex]=max(Candidatevalue);
%          upddatefunction(capitalindex,productivityindex)=bestvalue;
%          policyindex(capitalindex,productivityindex)=bestindex;
%          policycapital(capitalindex,productivityindex)=capitalgrid(bestindex);
%     end
% end

完整代码
for iter=1:p.maxiter
    updatefunction = zeros(capitalgridcount, productivitystatecount);

    for productivityindex=1:productivitystatecount
        currentproductivity=productivitystate(productivityindex);
        expected=valuefunction*transitionmatrix(productivityindex,:)';
        for capitalindex=1:capitalgridcount
            currentcapital=capitalgrid(capitalindex);
             y=currentproductivity*currentcapital^p.alpha;
             c=y+(1-p.delta)*currentcapital-capitalgrid;
             currentU=-inf(capitalgridcount,1);
             feasible=c>0;
             currentU(feasible)=log(c(feasible));%这是一种常用的逻辑索引，学习一下
             Candidatevalue=currentU+p.beta*expected;
             [bestvalue,bestindex]=max(Candidatevalue);
             updatefunction(capitalindex,productivityindex)=bestvalue;
             policyindex(capitalindex,productivityindex)=bestindex;
             policycapital(capitalindex,productivityindex)=capitalgrid(bestindex);
        end
    end
    diff=max(abs(updatefunction(:)-valuefunction(:)));
    valuefunction=updatefunction;
    if mod(iter, 100) == 0
        fprintf('iter = %d, diff = %.6f\n', iter, diff);
    end
    if diff<p.tol
        fprintf('收敛！迭代次数 = %d\n', iter);
        break;
    end
    if iter==p.maxiter
        fprintf('达到最大迭代次数，最终误差 = %.8f\n', diff);
        break;
    end
end



figure
plot(capitalgrid, valuefunction(:,1), 'LineWidth', 1.5)
hold on
plot(capitalgrid, valuefunction(:,2), 'LineWidth', 1.5)
grid on
legend('低生产率', '高生产率')
xlabel('资本 k')
ylabel('值函数 V(k,z)')
title('值函数')
plot(capitalgrid, policycapital(:,1))
plot(capitalgrid, policycapital(:,2))
plot(capitalgrid, capitalgrid, '--k')


