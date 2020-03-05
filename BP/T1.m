% 测试
clear
clc

%% 参数
global w            % 隐层神经元与输出层神经元之间的连接权
global v            % 输入层神经元与隐层神经元之间的连接权
global theta        % 输出层神经元阈值
global gama         % 隐层神经元阈值
%% 生成样本
XS = (1:100).*power(-1,randi(2,1,100));
YS = double(XS>0);
%% 调用学习主函数
Q = 1;
yeta = 0.1;
turns = 100;
[w_s, v_s, theta_s, gama_s]=main(XS,YS,Q,yeta,turns);
%% 测试
XST = (101:0.1:200).*power(-1,randi(2,1,991));
YST = double(XST>0);
YSN = zeros(1,length(XST));
for k = 1:length(XST)
    [~,~,~,YSN(k)] = neuralNet(XST(k));
end
score = YST - round(YSN);
find(score~=0)