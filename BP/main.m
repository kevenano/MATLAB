function [w_s, v_s, theta_s, gama_s]=main(XS,YS,Q,yeta,turns)
%MAIN ������
%   XS: ������������ֵ
%   YS: �����������ֵ
%   Q: ������Ԫ������
%   yeta: ѧϰ��
%   turns: ѭ������
%% ����ȫ�ֱ���
global w            % ������Ԫ���������Ԫ֮�������Ȩ
global v            % �������Ԫ��������Ԫ֮�������Ȩ
global theta        % �������Ԫ��ֵ
global gama         % ������Ԫ��ֵ

global l            % Y�ĳ��ȣ������������
global q            % ������Ԫ������
global d            % X�ĳ��ȣ�������������

l = length(YS(1));
q = Q;
d = length(XS(1));
%% ��ʼ������Ȩֵ����ֵ
w = rand(l,q);
v = rand(q,d);
theta = rand(1,l);
gama = rand(1,q);
%% ѭ��������������
m = length(XS);            % ��������
Ek = zeros(1, m);          % ���������ľ������
E = zeros(1, turns);       % �ۼ����
w_s = zeros(length(w), m);
v_s = zeros(length(v),m);
theta_s = zeros(length(theta),m);
gama_s = zeros(length(gama),m);
for turn = 1:turns
    w_s(:,turn) = w;
    v_s(:,turn) = v;
    theta_s(:,turn) = theta;
    gama_s(:,turn) = gama;
    for k = 1:m
        X = XS(k);                      % ��ǰ��������ֵ
        Y = YS(k);                      % ��ǰ�������ֵ
        Ek(k) = oneSample(X,Y,yeta);    % �����k������
    end
    E(turn) = 1/m*Ek*(Ek.');            % �����ۼ����
    sprintf('��%d�� �ۼ����:%.4f',turn,E(turn))
end
end

