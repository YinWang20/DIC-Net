%%
% A demo for generation image pair

% Load image as deformed one 
path1 = 'path';
def = imread(path1);

figure(1)
imshow(def)

% Transform image for uint8 to double
def = double(def);

% Get random Hermite displacement
element_size = 65;
[u_global, ux_global, uy_global, v_global, vx_global, vy_global] = Hermite2D(element_size);
u = u_global;
v = v_global;

% Applying displacement field to the reference image
[r, c] = size(def);
OD = -7;
[X, Y] = meshgrid(1:c, 1:r);
xdef = X + u;  
ydef = Y + v;
zref = bsp2(def, xdef, ydef, OD);


% Show disp and reference image
figure(2)
colormap('jet')
surf(X, Y, u);
shading interp
view(2)
axis equal
grid off

figure(3)
colormap('jet')
surf(X, Y, v);
shading interp
view(2)
axis equal
grid off

figure(4)
imshow(mat2gray(zref, [0,255]))