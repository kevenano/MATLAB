function Ek = argvUpdate(X, Y, Yn, b, yeta)
%ARGVUPDATE �������º���
%   Y: ʵ��ֵ
%   Yn: ����ֵ
%   b: ������Ԫ�����
%   yeta: ѧϰ��
%   Ek: �������

global w            % ������Ԫ���������Ԫ֮�������Ȩ
global v            % �������Ԫ��������Ԫ֮�������Ȩ
global theta        % �������Ԫ��ֵ
global gama         % ������Ԫ��ֵ

global q            % ������Ԫ������
global d            % X�ĳ��ȣ�������������

Ek = 0.5*(Yn-Y)*((Yn-Y).');                         % ��k�������ľ������
g = Yn.*(1-Yn).*(Y-Yn);                             % �������Ԫ���ݶ���
e = b.*(1-b).*(g*w);                                % ������Ԫ���ݶ���
delta_w = yeta*repmat(g.',1,q)*repmat(b.',1,q);     
delta_v = yeta*repmat(e.',1,d)*repmat(X.',1,d);
delta_theta = -1*yeta*g;
delta_gama = -1*yeta*e;

w = w+delta_w;
v = v+delta_v;
theta = theta+ delta_theta;
gama = gama+delta_gama;
end

