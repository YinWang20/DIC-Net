
function  [uu,uux,uuy,vv,vvx,vvy] = Shapefunction(uxydata, vxydata, a)
%% Title: "DIC-Net: Upgrade the performance of traditional DIC with Hermite dataset and convolution neural network"
% doi: https://doi.org/10.1016/j.optlaseng.2022.107278
% Author: Yin Wang (yin-wang20@mails.tsinghua.edu.cn) and Jiaqing Zhao (jqzhao@mail.tsinghua.edu.cn) 
% Latest Date  : 2022-10-01

%% Input: uxydata, vxydata, a
% uxydata and vxydata are 1x9 vectors which stated as nodal DOFs in our paper
% a = (element_size - 1)/2

%% output: uu,uux,uuy,vv,vvx,vvy
% generated random displacement and strain field in the current element

%1D c2 Hermite polynomial shape function
N1_0 = @(k)-(k-1)^3*(3*k^2+9*k+8)/16;               N1_0_k = @(k)- ((6*k + 9)*(k - 1)^3)/16 - (3*(k - 1)^2*(3*k^2 + 9*k + 8))/16;
N2_0 = @(k)(k+1)^3*(3*k^2-9*k+8)/16;                N2_0_k = @(k)((6*k - 9)*(k + 1)^3)/16 + (3*(k + 1)^2*(3*k^2 - 9*k + 8))/16;
N1_1 = @(k)-(k-1)^3*(3*k+5)*(k+1)/16;             N1_1_k = @(k) - (3*(k - 1)^3*(k + 1))/16 - ((3*k + 5)*(k - 1)^3)/16 - (3*(3*k + 5)*(k - 1)^2*(k + 1))/16;
N2_1 = @(k)-(k+1)^3*(3*k-5)*(k-1)/16;             N2_1_k = @(k)- (3*(k - 1)*(k + 1)^3)/16 - ((3*k - 5)*(k + 1)^3)/16 - (3*(3*k - 5)*(k - 1)*(k + 1)^2)/16;
N1_2 = @(k)-(k-1)^3*(k+1)^2/16;                     N1_2_k = @(k)- ((2*k + 2)*(k - 1)^3)/16 - (3*(k - 1)^2*(k + 1)^2)/16;
N2_2 = @(k)(k+1)^3*(k-1)^2/16;                      N2_2_k = @(k)((2*k - 2)*(k + 1)^3)/16 + (3*(k - 1)^2*(k + 1)^2)/16;


%2D c2 Hermite polynomial shape function
N1_00 = @(k,q)N1_0(k)*N1_0(q);N1_00_x = @(k,q)N1_0_k(k)*N1_0(q);N1_00_y = @(k,q)N1_0(k)*N1_0_k(q);%u     
N1_10 = @(k,q)N1_1(k)*N1_0(q);N1_10_x = @(k,q)N1_1_k(k)*N1_0(q);N1_10_y = @(k,q)N1_1(k)*N1_0_k(q);%ux
N1_01 = @(k,q)N1_0(k)*N1_1(q);N1_01_x = @(k,q)N1_0_k(k)*N1_1(q);N1_01_y = @(k,q)N1_0(k)*N1_1_k(q);%uy
N1_11 = @(k,q)N1_1(k)*N1_1(q);N1_11_x = @(k,q)N1_1_k(k)*N1_1(q);N1_11_y = @(k,q)N1_1(k)*N1_1_k(q);%uxy
N1_20 = @(k,q)N1_2(k)*N1_0(q);N1_20_x = @(k,q)N1_2_k(k)*N1_0(q);N1_20_y = @(k,q)N1_2(k)*N1_0_k(q);%uxx
N1_02 = @(k,q)N1_0(k)*N1_2(q);N1_02_x = @(k,q)N1_0_k(k)*N1_2(q);N1_02_y = @(k,q)N1_0(k)*N1_2_k(q);%uyy
N1_21 = @(k,q)N1_2(k)*N1_1(q);N1_21_x = @(k,q)N1_2_k(k)*N1_1(q);N1_21_y = @(k,q)N1_2(k)*N1_1_k(q);%uxxy
N1_12 = @(k,q)N1_1(k)*N1_2(q);N1_12_x = @(k,q)N1_1_k(k)*N1_2(q);N1_12_y = @(k,q)N1_1(k)*N1_2_k(q);%uxyy
N1_22 = @(k,q)N1_2(k)*N1_2(q);N1_22_x = @(k,q)N1_2_k(k)*N1_2(q);N1_22_y = @(k,q)N1_2(k)*N1_2_k(q);%uxxyy

N2_00 = @(k,q)N2_0(k)*N1_0(q);N2_00_x = @(k,q)N2_0_k(k)*N1_0(q);N2_00_y = @(k,q)N2_0(k)*N1_0_k(q);%u
N2_10 = @(k,q)N2_1(k)*N1_0(q);N2_10_x = @(k,q)N2_1_k(k)*N1_0(q);N2_10_y = @(k,q)N2_1(k)*N1_0_k(q);%ux
N2_01 = @(k,q)N2_0(k)*N1_1(q);N2_01_x = @(k,q)N2_0_k(k)*N1_1(q);N2_01_y = @(k,q)N2_0(k)*N1_1_k(q);%uy
N2_11 = @(k,q)N2_1(k)*N1_1(q);N2_11_x = @(k,q)N2_1_k(k)*N1_1(q);N2_11_y = @(k,q)N2_1(k)*N1_1_k(q);%uxy
N2_20 = @(k,q)N2_2(k)*N1_0(q);N2_20_x = @(k,q)N2_2_k(k)*N1_0(q);N2_20_y = @(k,q)N2_2(k)*N1_0_k(q);%uxx
N2_02 = @(k,q)N2_0(k)*N1_2(q);N2_02_x = @(k,q)N2_0_k(k)*N1_2(q);N2_02_y = @(k,q)N2_0(k)*N1_2_k(q);%uyy
N2_21 = @(k,q)N2_2(k)*N1_1(q);N2_21_x = @(k,q)N2_2_k(k)*N1_1(q);N2_21_y = @(k,q)N2_2(k)*N1_1_k(q);%uxxy
N2_12 = @(k,q)N2_1(k)*N1_2(q);N2_12_x = @(k,q)N2_1_k(k)*N1_2(q);N2_12_y = @(k,q)N2_1(k)*N1_2_k(q);%uxyy
N2_22 = @(k,q)N2_2(k)*N1_2(q);N2_22_x = @(k,q)N2_2_k(k)*N1_2(q);N2_22_y = @(k,q)N2_2(k)*N1_2_k(q);%uxxyy

N3_00 = @(k,q)N2_0(k)*N2_0(q);N3_00_x = @(k,q)N2_0_k(k)*N2_0(q);N3_00_y = @(k,q)N2_0(k)*N2_0_k(q);%u
N3_10 = @(k,q)N2_1(k)*N2_0(q);N3_10_x = @(k,q)N2_1_k(k)*N2_0(q);N3_10_y = @(k,q)N2_1(k)*N2_0_k(q);%ux
N3_01 = @(k,q)N2_0(k)*N2_1(q);N3_01_x = @(k,q)N2_0_k(k)*N2_1(q);N3_01_y = @(k,q)N2_0(k)*N2_1_k(q);%uy
N3_11 = @(k,q)N2_1(k)*N2_1(q);N3_11_x = @(k,q)N2_1_k(k)*N2_1(q);N3_11_y = @(k,q)N2_1(k)*N2_1_k(q);%uxy
N3_20 = @(k,q)N2_2(k)*N2_0(q);N3_20_x = @(k,q)N2_2_k(k)*N2_0(q);N3_20_y = @(k,q)N2_2(k)*N2_0_k(q);%uxx
N3_02 = @(k,q)N2_0(k)*N2_2(q);N3_02_x = @(k,q)N2_0_k(k)*N2_2(q);N3_02_y = @(k,q)N2_0(k)*N2_2_k(q);%uyy
N3_21 = @(k,q)N2_2(k)*N2_1(q);N3_21_x = @(k,q)N2_2_k(k)*N2_1(q);N3_21_y = @(k,q)N2_2(k)*N2_1_k(q);%uxxy
N3_12 = @(k,q)N2_1(k)*N2_2(q);N3_12_x = @(k,q)N2_1_k(k)*N2_2(q);N3_12_y = @(k,q)N2_1(k)*N2_2_k(q);%uxyy
N3_22 = @(k,q)N2_2(k)*N2_2(q);N3_22_x = @(k,q)N2_2_k(k)*N2_2(q);N3_22_y = @(k,q)N2_2(k)*N2_2_k(q);%uxxyy

N4_00 = @(k,q)N1_0(k)*N2_0(q);N4_00_x = @(k,q)N1_0_k(k)*N2_0(q);N4_00_y = @(k,q)N1_0(k)*N2_0_k(q);%u
N4_10 = @(k,q)N1_1(k)*N2_0(q);N4_10_x = @(k,q)N1_1_k(k)*N2_0(q);N4_10_y = @(k,q)N1_1(k)*N2_0_k(q);%ux
N4_01 = @(k,q)N1_0(k)*N2_1(q);N4_01_x = @(k,q)N1_0_k(k)*N2_1(q);N4_01_y = @(k,q)N1_0(k)*N2_1_k(q);%uy
N4_11 = @(k,q)N1_1(k)*N2_1(q);N4_11_x = @(k,q)N1_1_k(k)*N2_1(q);N4_11_y = @(k,q)N1_1(k)*N2_1_k(q);%uxy
N4_20 = @(k,q)N1_2(k)*N2_0(q);N4_20_x = @(k,q)N1_2_k(k)*N2_0(q);N4_20_y = @(k,q)N1_2(k)*N2_0_k(q);%uxx
N4_02 = @(k,q)N1_0(k)*N2_2(q);N4_02_x = @(k,q)N1_0_k(k)*N2_2(q);N4_02_y = @(k,q)N1_0(k)*N2_2_k(q);%uyy
N4_21 = @(k,q)N1_2(k)*N2_1(q);N4_21_x = @(k,q)N1_2_k(k)*N2_1(q);N4_21_y = @(k,q)N1_2(k)*N2_1_k(q);%uxxy
N4_12 = @(k,q)N1_1(k)*N2_2(q);N4_12_x = @(k,q)N1_1_k(k)*N2_2(q);N4_12_y = @(k,q)N1_1(k)*N2_2_k(q);%uxyy
N4_22 = @(k,q)N1_2(k)*N2_2(q);N4_22_x = @(k,q)N1_2_k(k)*N2_2(q);N4_22_y = @(k,q)N1_2(k)*N2_2_k(q);%uxxyy

% local displacement and strain in current element

u = @(k,q)N1_00(k,q)*uxydata(1) + N1_10(k,q)*uxydata(2) + N1_01(k,q)*uxydata(3) + N1_11(k,q)*uxydata(4) + N1_20(k,q)*uxydata(5) + N1_02(k,q)*uxydata(6) + N1_21(k,q)*uxydata(7) + N1_12(k,q)*uxydata(8) + N1_22(k,q)*uxydata(9)... 
    + N2_00(k,q)*uxydata(10) + N2_10(k,q)*uxydata(11) + N2_01(k,q)*uxydata(12) + N2_11(k,q)*uxydata(13) + N2_20(k,q)*uxydata(14) + N2_02(k,q)*uxydata(15) + N2_21(k,q)*uxydata(16) + N2_12(k,q)*uxydata(17) + N2_22(k,q)*uxydata(18)...
    + N3_00(k,q)*uxydata(19) + N3_10(k,q)*uxydata(20) + N3_01(k,q)*uxydata(21) + N3_11(k,q)*uxydata(22) + N3_20(k,q)*uxydata(23) + N3_02(k,q)*uxydata(24) + N3_21(k,q)*uxydata(25) + N3_12(k,q)*uxydata(26) + N3_22(k,q)*uxydata(27)...
    +N4_00(k,q)*uxydata(28) + N4_10(k,q)*uxydata(29) + N4_01(k,q)*uxydata(30) + N4_11(k,q)*uxydata(31) + N4_20(k,q)*uxydata(32) + N4_02(k,q)*uxydata(33) + N4_21(k,q)*uxydata(34) + N4_12(k,q)*uxydata(35) + N4_22(k,q)*uxydata(36);

v = @(k,q)N1_00(k,q)*vxydata(1) + N1_10(k,q)*vxydata(2) + N1_01(k,q)*vxydata(3) + N1_11(k,q)*vxydata(4) + N1_20(k,q)*vxydata(5) + N1_02(k,q)*vxydata(6) + N1_21(k,q)*vxydata(7) + N1_12(k,q)*vxydata(8) + N1_22(k,q)*vxydata(9)... 
    + N2_00(k,q)*vxydata(10) + N2_10(k,q)*vxydata(11) + N2_01(k,q)*vxydata(12) + N2_11(k,q)*vxydata(13) + N2_20(k,q)*vxydata(14) + N2_02(k,q)*vxydata(15) + N2_21(k,q)*vxydata(16) + N2_12(k,q)*vxydata(17) + N2_22(k,q)*vxydata(18)...
    + N3_00(k,q)*vxydata(19) + N3_10(k,q)*vxydata(20) + N3_01(k,q)*vxydata(21) + N3_11(k,q)*vxydata(22) + N3_20(k,q)*vxydata(23) + N3_02(k,q)*vxydata(24) + N3_21(k,q)*vxydata(25) + N3_12(k,q)*vxydata(26) + N3_22(k,q)*vxydata(27)...
    +N4_00(k,q)*vxydata(28) + N4_10(k,q)*vxydata(29) + N4_01(k,q)*vxydata(30) + N4_11(k,q)*vxydata(31) + N4_20(k,q)*vxydata(32) + N4_02(k,q)*vxydata(33) + N4_21(k,q)*vxydata(34) + N4_12(k,q)*vxydata(35) + N4_22(k,q)*vxydata(36);

ux = @(k,q)N1_00_x(k,q)*uxydata(1) + N1_10_x(k,q)*uxydata(2) + N1_01_x(k,q)*uxydata(3) + N1_11_x(k,q)*uxydata(4) + N1_20_x(k,q)*uxydata(5) + N1_02_x(k,q)*uxydata(6) + N1_21_x(k,q)*uxydata(7) + N1_12_x(k,q)*uxydata(8) + N1_22_x(k,q)*uxydata(9)... 
    + N2_00_x(k,q)*uxydata(10) + N2_10_x(k,q)*uxydata(11) + N2_01_x(k,q)*uxydata(12) + N2_11_x(k,q)*uxydata(13) + N2_20_x(k,q)*uxydata(14) + N2_02_x(k,q)*uxydata(15) + N2_21_x(k,q)*uxydata(16) + N2_12_x(k,q)*uxydata(17) + N2_22_x(k,q)*uxydata(18)...
    + N3_00_x(k,q)*uxydata(19) + N3_10_x(k,q)*uxydata(20) + N3_01_x(k,q)*uxydata(21) + N3_11_x(k,q)*uxydata(22) + N3_20_x(k,q)*uxydata(23) + N3_02_x(k,q)*uxydata(24) + N3_21_x(k,q)*uxydata(25) + N3_12_x(k,q)*uxydata(26) + N3_22_x(k,q)*uxydata(27)...
    +N4_00_x(k,q)*uxydata(28) + N4_10_x(k,q)*uxydata(29) + N4_01_x(k,q)*uxydata(30) + N4_11_x(k,q)*uxydata(31) + N4_20_x(k,q)*uxydata(32) + N4_02_x(k,q)*uxydata(33) + N4_21_x(k,q)*uxydata(34) + N4_12_x(k,q)*uxydata(35) + N4_22_x(k,q)*uxydata(36);

uy = @(k,q)N1_00_y(k,q)*uxydata(1) + N1_10_y(k,q)*uxydata(2) + N1_01_y(k,q)*uxydata(3) + N1_11_y(k,q)*uxydata(4) + N1_20_y(k,q)*uxydata(5) + N1_02_y(k,q)*uxydata(6) + N1_21_y(k,q)*uxydata(7) + N1_12_y(k,q)*uxydata(8) + N1_22_y(k,q)*uxydata(9)... 
    + N2_00_y(k,q)*uxydata(10) + N2_10_y(k,q)*uxydata(11) + N2_01_y(k,q)*uxydata(12) + N2_11_y(k,q)*uxydata(13) + N2_20_y(k,q)*uxydata(14) + N2_02_y(k,q)*uxydata(15) + N2_21_y(k,q)*uxydata(16) + N2_12_y(k,q)*uxydata(17) + N2_22_y(k,q)*uxydata(18)...
    + N3_00_y(k,q)*uxydata(19) + N3_10_y(k,q)*uxydata(20) + N3_01_y(k,q)*uxydata(21) + N3_11_y(k,q)*uxydata(22) + N3_20_y(k,q)*uxydata(23) + N3_02_y(k,q)*uxydata(24) + N3_21_y(k,q)*uxydata(25) + N3_12_y(k,q)*uxydata(26) + N3_22_y(k,q)*uxydata(27)...
    +N4_00_y(k,q)*uxydata(28) + N4_10_y(k,q)*uxydata(29) + N4_01_y(k,q)*uxydata(30) + N4_11_y(k,q)*uxydata(31) + N4_20_y(k,q)*uxydata(32) + N4_02_y(k,q)*uxydata(33) + N4_21_y(k,q)*uxydata(34) + N4_12_y(k,q)*uxydata(35) + N4_22_y(k,q)*uxydata(36);

vx = @(k,q)N1_00_x(k,q)*vxydata(1) + N1_10_x(k,q)*vxydata(2) + N1_01_x(k,q)*vxydata(3) + N1_11_x(k,q)*vxydata(4) + N1_20_x(k,q)*vxydata(5) + N1_02_x(k,q)*vxydata(6) + N1_21_x(k,q)*vxydata(7) + N1_12_x(k,q)*vxydata(8) + N1_22_x(k,q)*vxydata(9)... 
    + N2_00_x(k,q)*vxydata(10) + N2_10_x(k,q)*vxydata(11) + N2_01_x(k,q)*vxydata(12) + N2_11_x(k,q)*vxydata(13) + N2_20_x(k,q)*vxydata(14) + N2_02_x(k,q)*vxydata(15) + N2_21_x(k,q)*vxydata(16) + N2_12_x(k,q)*vxydata(17) + N2_22_x(k,q)*vxydata(18)...
    + N3_00_x(k,q)*vxydata(19) + N3_10_x(k,q)*vxydata(20) + N3_01_x(k,q)*vxydata(21) + N3_11_x(k,q)*vxydata(22) + N3_20_x(k,q)*vxydata(23) + N3_02_x(k,q)*vxydata(24) + N3_21_x(k,q)*vxydata(25) + N3_12_x(k,q)*vxydata(26) + N3_22_x(k,q)*vxydata(27)...
    +N4_00_x(k,q)*vxydata(28) + N4_10_x(k,q)*vxydata(29) + N4_01_x(k,q)*vxydata(30) + N4_11_x(k,q)*vxydata(31) + N4_20_x(k,q)*vxydata(32) + N4_02_x(k,q)*vxydata(33) + N4_21_x(k,q)*vxydata(34) + N4_12_x(k,q)*vxydata(35) + N4_22_x(k,q)*vxydata(36);

vy = @(k,q)N1_00_y(k,q)*vxydata(1) + N1_10_y(k,q)*vxydata(2) + N1_01_y(k,q)*vxydata(3) + N1_11_y(k,q)*vxydata(4) + N1_20_y(k,q)*vxydata(5) + N1_02_y(k,q)*vxydata(6) + N1_21_y(k,q)*vxydata(7) + N1_12_y(k,q)*vxydata(8) + N1_22_y(k,q)*vxydata(9)... 
    + N2_00_y(k,q)*vxydata(10) + N2_10_y(k,q)*vxydata(11) + N2_01_y(k,q)*vxydata(12) + N2_11_y(k,q)*vxydata(13) + N2_20_y(k,q)*vxydata(14) + N2_02_y(k,q)*vxydata(15) + N2_21_y(k,q)*vxydata(16) + N2_12_y(k,q)*vxydata(17) + N2_22_y(k,q)*vxydata(18)...
    + N3_00_y(k,q)*vxydata(19) + N3_10_y(k,q)*vxydata(20) + N3_01_y(k,q)*vxydata(21) + N3_11_y(k,q)*vxydata(22) + N3_20_y(k,q)*vxydata(23) + N3_02_y(k,q)*vxydata(24) + N3_21_y(k,q)*vxydata(25) + N3_12_y(k,q)*vxydata(26) + N3_22_y(k,q)*vxydata(27)...
    +N4_00_y(k,q)*vxydata(28) + N4_10_y(k,q)*vxydata(29) + N4_01_y(k,q)*vxydata(30) + N4_11_y(k,q)*vxydata(31) + N4_20_y(k,q)*vxydata(32) + N4_02_y(k,q)*vxydata(33) + N4_21_y(k,q)*vxydata(34) + N4_12_y(k,q)*vxydata(35) + N4_22_y(k,q)*vxydata(36);

uu = zeros(2*a+1,2*a+1);
uux = zeros(2*a+1,2*a+1);
uuy = zeros(2*a+1,2*a+1);
vv = zeros(2*a+1,2*a+1);
vvx = zeros(2*a+1,2*a+1);
vvy = zeros(2*a+1,2*a+1);
for x = -a:1:a
    for y = -a:1:a
        uu(a+1-y,x+a+1) = u(x/a,y/a);
        uux(a+1-y,x+a+1) = 1/a*ux(x/a,y/a);
        uuy(a+1-y,x+a+1) = -1/a*uy(x/a,y/a);
        vv(a+1-y,x+a+1) = v(x/a,y/a);
        vvx(a+1-y,x+a+1) = 1/a*vx(x/a,y/a);
        vvy(a+1-y,x+a+1) = -1/a*vy(x/a,y/a);
    end
end



% u(1,-1)
% ux(1,-1)
% uy(1,-1)

% v(-1,1)
% vx(-1,1)
% vy(-1,1)

