function [alpha,b,beta,Yn] = neuralNet(X)
%NEURALNET ������
%   ����X������������������
%   alpha: ������Ԫ������
%   b:  ������Ԫ�����
%   beta: �������Ԫ������
%   Yn: �������Ԫ�����

global w            % ������Ԫ���������Ԫ֮�������Ȩ
global v            % �������Ԫ��������Ԫ֮�������Ȩ
global theta        % �������Ԫ��ֵ
global gama         % ������Ԫ��ֵ

alpha = X*(v.');                % ������Ԫ������
b = logsig(alpha - gama);       % ������Ԫ�����
beta = b*(w.');                 % �������Ԫ������
Yn = logsig(beta - theta);      % �������Ԫ�����
end

