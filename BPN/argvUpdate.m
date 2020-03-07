function Ek = argvUpdate(X,Y,Y_C,b,yeta)
%ARGVUPDATE 参数更新函数
%   输入：
%       X: 输入量
%       Y: 输入量对应输出
%       Y_C: 计算值
%       b: 各层输出
%       yeta: 学习率
%   返回：
%       Ek： 当前样本的均方误差

%% 声明全局变量
global v        % 输入层的权值
global w        % 各隐层的权值
global theta    % 各层阈值（隐层+输出层）

global l        % 各层节点数（隐层+输出层）
global n        % 隐层层数
%% 
e = Y_C.*(1-Y_C).*(Y-Y_C);              % 输出层梯度项
delta_theta = -1*yeta*e;                % 输出层阈值更新量
theta{n+1} = theta{n+1}+delta_theta;    % 更新输出层阈值
for layer = n:-1:1
    delta_w = yeta*repmat(e.',1,l(layer)).*repmat(b{layer},l(layer+1),1);       % 隐层权值更新量
    w{layer} = w{layer} + delta_w;                                              % 更新隐层权值
    e = e*w{layer}.*b{layer}.*(1-b{layer});                                     % 更新梯度项
    delta_theta = -1*yeta*e;                                                    % 隐层层阈值更新量
    theta{layer} = theta{layer} + delta_theta;                                  % 更新隐层阈值    
end
delta_v = yeta*repmat(e.',1,length(X)).*repmat(X,l(1),1);       % 输入层权值更新量
v = v+delta_v;                                             % 跟新输入层权值
Ek = 0.5*(Y-Y_C)*((Y-Y_C).');                              % 当前样本的均方误差
end

