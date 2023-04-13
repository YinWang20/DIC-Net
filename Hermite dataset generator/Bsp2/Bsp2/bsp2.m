function  [ZI C] = bsp2 (A, XI, YI, od)
% 任意次B样条插值，比bsp插值阶次多，bsp只能3次样条插值
% 指定二维double图像A, 和待插值点的XI和YI坐标（坐标从1开始，到维数
% 【dll内部坐标从0开始，本程序按matlab风格调用】）
% od 为插值阶次，{1,3,4,5,7, -2,-3,-4,-5,-6,-7,-8,-9}可选
%    其中 负数表示用原始程序，正数表示用我修改后的更快的程序
% 返回插值ZI及B样条系数C.
% 注意：XI,YI不能越界，否则结果不正确。
% dll工程来源 E:\WORKSPACE\DResearch\HSPD\FCfiles\Official\bspfull

% 注意 ： 32位的matlab必须调用32位的dll
%         64位的matlab必须调用64位的dll


if nargin<3
    error('Not enough input args.');
end
if nargin<4, od = 3; end

if all(od ~= [1, 3, 4, 5, 7, -2,-3,-4,-5,-6,-7,-8,-9]) 
   error('插值阶次必须是 [1, 3, 4, 5, 7, -2,-3,-4,-5,-6,-7,-8,-9] 中的一个！'); 
end

BSPFULL = ['bspfull' num2str(getmatlabversion())];


if ~libisloaded(BSPFULL)
    loadlibrary(BSPFULL,'bspfull.h');
    %libfunctions bspfull -full %用来查看接口
end

[n,m]=size(A); 
A = A'; A = A(:); % 很关键：图像按行，写成向量传给bspfast
xp = libpointer('doublePtr',A);
calllib(BSPFULL, 'SamplesToCoefficients', xp, m,n ,od);

% 传入dll坐标从0开始
XI = XI - 1.0; YI = YI - 1.0;
SX = size(XI);
XI = XI(:); YI = YI(:);
ZI = XI;

xsp = libpointer('doublePtr',XI); 
ysp = libpointer('doublePtr',YI); 
zsp = libpointer('doublePtr',ZI); 
calllib(BSPFULL, 'interpfull', xp, m,n, xsp,ysp,zsp, numel(XI),od);
ZI = zsp.value;

% 变成和原始XI相同的形式
ZI = reshape(ZI, SX);


if nargout > 1
   C = xp.value;
end


%  if libisloaded('bspfull')
%      unloadlibrary('bspfull');
%  end



%{
% bsp2和matalb的三次样条插值相同！以下是验证：
% 平均误差1e-010,最大误差1e-7
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


