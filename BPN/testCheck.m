function [E,YSN] = testCheck(XST,YST,v,w,theta,n)
%TEST 测试验证函数
%   输入：
%       v：输入层的权值
%       w：各隐层的权值
%       theta：各层阈值(隐层+输出层)
%       n: 隐层数量
%   返回累计误差
%%
YSN = zeros(1,length(XST));
Ek = zeros(1,length(XST));
for k = 1:length(XST)
    [YSN(k),~] = neuroNet(XST{k},v,w,theta,n);
    Ek(k) = 0.5*(YSN(k)-YST{k})*((YSN(k)-YST{k}).');    % 第k个验证样本的均方误差
end
E = (1/length(Ek))*Ek*(Ek.');     % 累计误差
end

