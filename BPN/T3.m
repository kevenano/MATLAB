% 测试
close all
clear
clc

%% 参数
global v        % 输入层的权值
global w        % 各隐层的权值
global theta    % 各层阈值（隐层+输出层）

global l        % 各层节点数（隐层+输出层）（一维行向量）
global n        % 隐层层数
%% 生成样本
x = 0:0.01:6;
y = 0:0.01:6;
[Xx,Xy] = meshgrid(x,y);
Y = int8(((Xx-3).^2+(Xy-3).^2)<=1);
XT = zeros(length(x)*length(y), 2);
YT = zeros(length(x)*length(y), 1);
count = 1;
for i=1:length(x)
    for j=1:length(y)
        XT(count,:) = [Xx(i,j) Xy(i,j)];
        YT(count,:) = Y(i,j);
        count = count + 1;
    end
end
% 训练样本
tt =randi(length(XT),1,20000);
Xs = XT(tt,:);
Ys = YT(tt,:);
XS = num2cell(Xs,2);
YS = num2cell(Ys,2);
% 测试样本
ts = 1:length(XT);
ts(tt)=[];
ts = randi(length(ts),1,10000);
Xst = XT(ts,:);
Yst = YT(ts,:);
XST = num2cell(Xst,2);
YST = num2cell(Yst,2);
%% 调用学习主函数
tic
l = [8 8 length(YS(1))];
n = length(l)-1;
yeta = 0.1;
turns = 5000;
[E, Et]=main(XS,YS,yeta,turns,XST,YST);
fileName = datestr(now,'yy-mm-dd-HH-MM-SS');
figure
plot(1:turns,E,1:turns,Et);
xlabel('Turn')
ylabel('E/Et')
legend('E','Et')
saveas(gcf,join([fileName,'_E']),'fig');
toc
%%
v_t = v;
w_t = w;
theta_t = theta;
n_t = n;
[~,YSN] = testCheck(XST,YST,v_t,w_t,theta_t,n_t);  % 计算测试集的累计误差
figure
scatter3(Xst(:,1),Xst(:,2),YSN);
axis equal
saveas(gcf,join([fileName,'_3YSN']),'fig');
figure 
scatter(Xst(:,1),Xst(:,2),[],Yst)
title('Yst')
axis equal
saveas(gcf,join([fileName,'_2Yst']),'fig');
figure 
scatter(Xst(:,1),Xst(:,2),[],YSN)
title('YSN')
axis equal
saveas(gcf,join([fileName,'_2YSN']),'fig');
figure 
scatter(Xst(:,1),Xst(:,2),[],round(YSN))
title('round(YSN)')
axis equal
saveas(gcf,join([fileName,'_rYSN']),'fig');
save(fileName)