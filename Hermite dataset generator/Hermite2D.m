function [u_global, ux_global, uy_global, v_global, vx_global, vy_global] = Hermite2D(element_size)

%% Title: "DIC-Net: Upgrade the performance of traditional DIC with Hermite dataset and convolution neural network"
% doi: https://doi.org/10.1016/j.optlaseng.2022.107278
% Author: Yin Wang (yin-wang20@mails.tsinghua.edu.cn) and Jiaqing Zhao (jqzhao@mail.tsinghua.edu.cn) 
% Latest Date  : 2022-10-01

%% Introduction of Hermite2D
%
% Function Hermite2D() receives a element_size which you select from [5, 9,
% 17, 33, 65] and splits the image use Hermite element with this size, the
% output is random full field disp and strain of the image
% 

%% Parameters in function correponding to Table 1 in paper
%
% disp/2 is equal to the magnitude of the Displacement in paper
%
% udata_all_node.u,udata_all_node.ux and udata_all_node.uy corresponding to
% the disp and strain range in paper, so as the v

%% Extension
% If you want to define your own element_size and image size, you should only
% change the corresponding parameter in the switch case function body, and
% the u_global,ux_global,uy_global (so as v) size.
switch element_size
    case 5
        disp = 1;
        node = 49;
        a = 2;
        delta = 4;
    case 9
        disp = 1;
        node = 25;
        a = 4;
        delta = 8;
    case 17
        disp = 2;
        node = 13;
        a = 8;
        delta = 16;
    case 33
        disp = 3;
        node = 7;
        a = 16;
        delta = 32;
    case 65
        disp = 4;
        node = 4;
        a = 32;
        delta = 64;
end

u_global = zeros(193);
ux_global = zeros(193);
uy_global = zeros(193);
v_global = zeros(193);
vx_global = zeros(193);
vy_global = zeros(193);

udata_all_node = struct();
udata_all_node.u = disp*rand(node)-disp/2;
udata_all_node.ux = 0.06*rand(node)-0.03;
udata_all_node.uy = 0.06*rand(node)-0.03;

vdata_all_node = struct();
vdata_all_node.v = disp*rand(node)-disp/2;
vdata_all_node.vx = 0.06*rand(node)-0.03;
vdata_all_node.vy = 0.06*rand(node)-0.03;

for i = 1:node-1
    for j = 1:node-1
        uxydata = [udata_all_node.u(i+1,j) udata_all_node.ux(i+1,j) udata_all_node.uy(i+1,j),0.002*rand-0.001,0.002*rand-0.001,0.002*rand-0.001,0.0002*rand-0.0001,0.0002*rand-0.0001,0.00002*rand-0.00001,...
            udata_all_node.u(i+1,j+1) udata_all_node.ux(i+1,j+1) udata_all_node.uy(i+1,j+1),0.002*rand-0.001,0.002*rand-0.001,0.002*rand-0.001,0.0002*rand-0.0001,0.0002*rand-0.0001,0.00002*rand-0.00001,...
            udata_all_node.u(i,j+1) udata_all_node.ux(i,j+1) udata_all_node.uy(i,j+1),0.002*rand-0.001,0.002*rand-0.001,0.002*rand-0.001,0.0002*rand-0.0001,0.0002*rand-0.0001,0.00002*rand-0.00001,...
            udata_all_node.u(i,j) udata_all_node.ux(i,j) udata_all_node.uy(i,j),0.002*rand-0.001,0.002*rand-0.001,0.002*rand-0.001,0.0002*rand-0.0001,0.0002*rand-0.0001,0.00002*rand-0.00001,];
        
        vxydata = [vdata_all_node.v(i+1,j) vdata_all_node.vx(i+1,j) vdata_all_node.vy(i+1,j),0.002*rand-0.001,0.002*rand-0.001,0.002*rand-0.001,0.0002*rand-0.0001,0.0002*rand-0.0001,0.00002*rand-0.00001,...
            vdata_all_node.v(i+1,j+1) vdata_all_node.vx(i+1,j+1) vdata_all_node.vy(i+1,j+1),0.002*rand-0.001,0.002*rand-0.001,0.002*rand-0.001,0.0002*rand-0.0001,0.0002*rand-0.0001,0.00002*rand-0.00001,...
            vdata_all_node.v(i,j+1) vdata_all_node.vx(i,j+1) vdata_all_node.vy(i,j+1),0.002*rand-0.001,0.002*rand-0.001,0.002*rand-0.001,0.0002*rand-0.0001,0.0002*rand-0.0001,0.00002*rand-0.00001,...
            vdata_all_node.v(i,j) vdata_all_node.vx(i,j) vdata_all_node.vy(i,j),0.002*rand-0.001,0.002*rand-0.001,0.002*rand-0.001,0.0002*rand-0.0001,0.0002*rand-0.0001,0.00002*rand-0.00001,];
        
        [uu,uux,uuy,vv,vvx,vvy] = Shapefunction(uxydata, vxydata, a);
        u_global(1+(i-1)*delta:1+delta*i,1+(j-1)*delta:1+delta*j) = uu;
        ux_global(1+(i-1)*delta:1+delta*i,1+(j-1)*delta:1+delta*j) = uux;
        uy_global(1+(i-1)*delta:1+delta*i,1+(j-1)*delta:1+delta*j) = uuy;
        
        v_global(1+(i-1)*delta:1+delta*i,1+(j-1)*delta:1+delta*j) = vv;
        vx_global(1+(i-1)*delta:1+delta*i,1+(j-1)*delta:1+delta*j) = vvx;
        vy_global(1+(i-1)*delta:1+delta*i,1+(j-1)*delta:1+delta*j) = vvy;
    end
end


