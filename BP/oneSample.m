function Ek = oneSample(X,Y,yeta)
%ONESAMPLE ��������������
%   yeta: ѧϰ��

%%
[~,b,~,Yn] = neuralNet(X);               % ��������������
Ek = argvUpdate(X, Y, Yn, b, yeta);    % ���²���, ͬʱ����������
end

