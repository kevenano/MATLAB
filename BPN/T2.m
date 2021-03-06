% 测试
clear
clc

%% 参数
global v        % 输入层的权值
global w        % 各隐层的权值
global theta    % 各层阈值（隐层+输出层）

global l        % 各层节点数（隐层+输出层）（一维行向量）
global n        % 隐层层数
%% 生成样本
XS = (1:100).*power(-1,randi(2,1,100));
YS = double(XS>0);
%% 调用学习主函数
l = [16 length(YS(1))];
n = length(l)-1;
yeta = 0.1;
turns = 5000;
main(XS,YS,yeta,turns);
%% 测试
v_t = v;
w_t = w;
theta_t = theta;
n_t = n;

XST = (101:0.1:200).*power(-1,randi(2,1,991));
YST = double(XST>0);
YSN = zeros(1,length(XST));
for k = 1:length(XST)
    [YSN(k),~] = neuroNet(XST(k),v_t,w_t,theta_t,n_t);
end
score = YST - round(YSN);
find(score~=0)