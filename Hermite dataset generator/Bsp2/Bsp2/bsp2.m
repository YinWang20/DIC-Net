function  [ZI C] = bsp2 (A, XI, YI, od)
% �����B������ֵ����bsp��ֵ�״ζ࣬bspֻ��3��������ֵ
% ָ����άdoubleͼ��A, �ʹ���ֵ���XI��YI���꣨�����1��ʼ����ά��
% ��dll�ڲ������0��ʼ��������matlab�����á���
% od Ϊ��ֵ�״Σ�{1,3,4,5,7, -2,-3,-4,-5,-6,-7,-8,-9}��ѡ
%    ���� ������ʾ��ԭʼ����������ʾ�����޸ĺ�ĸ���ĳ���
% ���ز�ֵZI��B����ϵ��C.
% ע�⣺XI,YI����Խ�磬����������ȷ��
% dll������Դ E:\WORKSPACE\DResearch\HSPD\FCfiles\Official\bspfull

% ע�� �� 32λ��matlab�������32λ��dll
%         64λ��matlab�������64λ��dll


if nargin<3
    error('Not enough input args.');
end
if nargin<4, od = 3; end

if all(od ~= [1, 3, 4, 5, 7, -2,-3,-4,-5,-6,-7,-8,-9]) 
   error('��ֵ�״α����� [1, 3, 4, 5, 7, -2,-3,-4,-5,-6,-7,-8,-9] �е�һ����'); 
end

BSPFULL = ['bspfull' num2str(getmatlabversion())];


if ~libisloaded(BSPFULL)
    loadlibrary(BSPFULL,'bspfull.h');
    %libfunctions bspfull -full %�����鿴�ӿ�
end

[n,m]=size(A); 
A = A'; A = A(:); % �ܹؼ���ͼ���У�д����������bspfast
xp = libpointer('doublePtr',A);
calllib(BSPFULL, 'SamplesToCoefficients', xp, m,n ,od);

% ����dll�����0��ʼ
XI = XI - 1.0; YI = YI - 1.0;
SX = size(XI);
XI = XI(:); YI = YI(:);
ZI = XI;

xsp = libpointer('doublePtr',XI); 
ysp = libpointer('doublePtr',YI); 
zsp = libpointer('doublePtr',ZI); 
calllib(BSPFULL, 'interpfull', xp, m,n, xsp,ysp,zsp, numel(XI),od);
ZI = zsp.value;

% ��ɺ�ԭʼXI��ͬ����ʽ
ZI = reshape(ZI, SX);


if nargout > 1
   C = xp.value;
end


%  if libisloaded('bspfull')
%      unloadlibrary('bspfull');
%  end



%{
% bsp2��matalb������������ֵ��ͬ����������֤��
% ƽ�����1e-010,������1e-7
clc
A = imread2('bsp2matlabVERY.bmp');
[r, c] = size(A);
[X Y] = meshgrid(1:c, 1:r);
[XI YI] = meshgrid(10:c-10, 10:r-10);

% bsp2
od = 3; XINT = XI+0.35; YINT = YI+0.7642;
ZI = bsp2 (A, XINT, YINT, od);
% MATLAB
ZJ = interp2(X, Y, A, XINT, YINT, 'spline');

figure(43);
e=ZI - ZJ; %A(11:r-9, 11:c-9);
surf(e);
rg(e)
% min= -2.5654e-07 max= 9.574e-07 std=2.035e-08 mean=2.4387e-10



%}


