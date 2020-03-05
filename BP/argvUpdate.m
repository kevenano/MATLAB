function Ek = argvUpdate(X, Y, Yn, b, yeta)
%ARGVUPDATE 参数更新函数
%   Y: 实际值
%   Yn: 计算值
%   b: 隐层神经元的输出
%   yeta: 学习率
%   Ek: 均方误差

global w            % 隐层神经元与输出层神经元之间的连接权
global v            % 输入层神经元与隐层神经元之间的连接权
global theta        % 输出层神经元阈值
global gama         % 隐层神经元阈值

global q            % 隐层神经元的数量
global d            % X的长度（输入属性数）

Ek = 0.5*(Yn-Y)*((Yn-Y).');                         % 第k个样本的均方误差
g = Yn.*(1-Yn).*(Y-Yn);                             % 输出层神经元的梯度项
e = b.*(1-b).*(g*w);                                % 隐层神经元的梯度项
delta_w = yeta*repmat(g.',1,q)*repmat(b.',1,q);     
delta_v = yeta*repmat(e.',1,d)*repmat(X.',1,d);
delta_theta = -1*yeta*g;
delta_gama = -1*yeta*e;

w = w+delta_w;
v = v+delta_v;
theta = theta+ delta_theta;
gama = gama+delta_gama;
end

