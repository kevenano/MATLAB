function [Y_C,b] = neuroNet(X,v,w,theta,n)
%NEURONET 神经网络
%   输入：
%       X: 输入样本
%       v：输入层的权值
%       w：各隐层的权值
%       theta：各层阈值(隐层+输出层)
%       n: 隐层数量
%   返回：
%       Y_C: Y的计算值
%       b: 各层的输出（隐层+输出层）

%%
b = cell(1,n+1);                % 初始化各层输出
beta = X*(v.');                 % 第1隐层的输入
b{1} = logsig(beta - theta{1}); % 第1隐层的输出
for layer = 2:(n+1)
    beta = b{layer-1}*(w{layer-1}.');
    b{layer} = logsig(beta - theta{layer});
end
Y_C = b{n+1};       % 第n+1层即输出层
end

