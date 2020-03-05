% ����
clear
clc

%% ����
global w            % ������Ԫ���������Ԫ֮�������Ȩ
global v            % �������Ԫ��������Ԫ֮�������Ȩ
global theta        % �������Ԫ��ֵ
global gama         % ������Ԫ��ֵ
%% ��������
XS = (1:100).*power(-1,randi(2,1,100));
YS = double(XS>0);
%% ����ѧϰ������
Q = 1;
yeta = 0.1;
turns = 100;
[w_s, v_s, theta_s, gama_s]=main(XS,YS,Q,yeta,turns);
%% ����
XST = (101:0.1:200).*power(-1,randi(2,1,991));
YST = double(XST>0);
YSN = zeros(1,length(XST));
for k = 1:length(XST)
    [~,~,~,YSN(k)] = neuralNet(XST(k));
end
score = YST - round(YSN);
find(score~=0)