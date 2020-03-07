function main(XS,YS,yeta,turns)
%MAIN 训练主函数
%   XS: 所有样本输入值
%   YS: 所有样本输出值
%   yeta: 学习率
%   turns: 循环轮数
%   注意：权值矩阵，第一项表示+1层的节点，第二项表示本层节点
%% 声明全局变量
global v        % 输入层的权值
global w        % 各隐层的权值
global theta    % 各层阈值（隐层+输出层）

global l        % 各层节点数（隐层+输出层）（一维行向量）
global n        % 隐层层数
%% 初始化所有权值和阈值
v = rand(l(1),length(XS(1)));
w = cell(1,n);
theta = cell(1,n+1);
for layer=1:n
    w{layer} = rand(l(layer+1),l(layer));
    theta{layer} = rand(1,l(layer));
end
theta{n+1} = rand(1,l(n+1));
%% 循环处理所有样本
m = length(XS);            % 样本总数
Ek = zeros(1, m);          % 所有样本的均方误差
E = zeros(1, turns);       % 累计误差
for turn = 1:turns
    for k = 1:m
        X = XS(k);                          % 当前样本输入值
        Y = YS(k);                          % 当前样本输出值
        Ek(k) = singleProcess(X,Y,yeta);    % 处理第k个样本
    end
    E(turn) = 1/m*Ek*(Ek.');                % 计算累计误差
    sprintf('Turns %d, E:%.10f',turn,E(turn))
end
end

