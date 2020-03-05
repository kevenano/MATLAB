function Ek = oneSample(X,Y,yeta)
%ONESAMPLE 单个样本处理函数
%   yeta: 学习率

%%
[~,b,~,Yn] = neuralNet(X);               % 进入神经网络运算
Ek = argvUpdate(X, Y, Yn, b, yeta);    % 更新参数, 同时计算均方误差
end

