function [w_s, v_s, theta_s, gama_s]=main(XS,YS,Q,yeta,turns)
%MAIN 主函数
%   XS: 所有样本输入值
%   YS: 所有样本输出值
%   Q: 隐层神经元的数量
%   yeta: 学习率
%   turns: 循环轮数
%% 定义全局变量
global w            % 隐层神经元与输出层神经元之间的连接权
global v            % 输入层神经元与隐层神经元之间的连接权
global theta        % 输出层神经元阈值
global gama         % 隐层神经元阈值

global l            % Y的长度（输出属性数）
global q            % 隐层神经元的数量
global d            % X的长度（输入属性数）

l = length(YS(1));
q = Q;
d = length(XS(1));
%% 初始化所有权值和阈值
w = rand(l,q);
v = rand(q,d);
theta = rand(1,l);
gama = rand(1,q);
%% 循环处理所有样本
m = length(XS);            % 样本总数
Ek = zeros(1, m);          % 所有样本的均方误差
E = zeros(1, turns);       % 累计误差
w_s = zeros(length(w), m);
v_s = zeros(length(v),m);
theta_s = zeros(length(theta),m);
gama_s = zeros(length(gama),m);
for turn = 1:turns
    w_s(:,turn) = w;
    v_s(:,turn) = v;
    theta_s(:,turn) = theta;
    gama_s(:,turn) = gama;
    for k = 1:m
        X = XS(k);                      % 当前样本输入值
        Y = YS(k);                      % 当前样本输出值
        Ek(k) = oneSample(X,Y,yeta);    % 处理第k个样本
    end
    E(turn) = 1/m*Ek*(Ek.');            % 计算累计误差
    sprintf('第%d轮 累计误差:%.4f',turn,E(turn))
end
end

