function [alpha,b,beta,Yn] = neuralNet(X)
%NEURALNET 神经网络
%   输入X行向量（单个样本）
%   alpha: 隐层神经元的输入
%   b:  隐层神经元的输出
%   beta: 输出层神经元的输入
%   Yn: 输出层神经元的输出

global w            % 隐层神经元与输出层神经元之间的连接权
global v            % 输入层神经元与隐层神经元之间的连接权
global theta        % 输出层神经元阈值
global gama         % 隐层神经元阈值

alpha = X*(v.');                % 隐层神经元的输入
b = logsig(alpha - gama);       % 隐层神经元的输出
beta = b*(w.');                 % 输出层神经元的输入
Yn = logsig(beta - theta);      % 输出层神经元的输出
end

