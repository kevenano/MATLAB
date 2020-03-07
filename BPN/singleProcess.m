function Ek = singleProcess(X,Y,yeta)
%SINGLEPROCESS 单个样本处理函数
%   yeta: 学习率

%% 声明全局变量
global v        % 输入层的权值
global w        % 各隐层的权值
global theta    % 各层阈值（隐层+输出层）

% global l        % 各层节点数（隐层+输出层）
global n        % 隐层层数
%%
[Y_C,b] = neuroNet(X,v,w,theta,n);      % 进入神经网络计算
Ek = argvUpdate(X,Y,Y_C,b,yeta);        % 更新参数同时取得均方误差
end

