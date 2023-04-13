
%% Title: "DIC-Net: Upgrade the performance of traditional DIC with Hermite dataset and convolution neural network"
% doi: https://doi.org/10.1016/j.optlaseng.2022.107278
% Author: Yin Wang (yin-wang20@mails.tsinghua.edu.cn) and Jiaqing Zhao (jqzhao@mail.tsinghua.edu.cn) 
% Latest Date  : 2022-10-24

%%
clc
clear all
mkdir Train_Data
mkdir Test_Data

for i = 1:429
    for j = 1:100
        if j<=20
            element_size = 5;
        elseif j<=40 && j>20
            element_size = 9;
        elseif j<=60 && j>40
            element_size = 17;
        elseif j<=80 && j>60
            element_size = 33;
        elseif j<=100 && j>80
            element_size = 65;
        end
        
        %Load images
        directory = 'path'; % The directory where the input images are saved
        
        x = randi([1,66]);
        y = randi([1,66]);
        img = imread([directory, int2str(i),'.bmp']);
        img = double(img);
    
        %Displacement generation using Hermite element
        [u_global, ux_global, uy_global, v_global, vx_global, vy_global] = Hermite2D(element_size);
        u = u_global;
        v = v_global;
    
        %Applying displacement field to the reference image
        [r, c] = size(img);
        OD = -7;
        [X, Y] = meshgrid(1:c, 1:r);
        xdef = X + u;  
        ydef = Y + v;
        zref = bsp2(img, xdef, ydef, OD);
    
        %Obtain cropped field, as well as reference and deformation images
        def_float = zref(x:x+127,y:y+127);
        ref_float = img(x:x+127,y:y+127);
    
        uu(1,:,:) = u_global(x:x+127,y:y+127);
        uu(2,:,:) = v_global(x:x+127,y:y+127);
    
        E(1,:,:) = ux_global(x:x+127,y:y+127);
        E(2,:,:) = vy_global(x:x+127,y:y+127);
        E(3,:,:) = 0.5*(uy_global(x:x+127,y:y+127)+vx_global(x:x+127,y:y+127));
    
        % Adding Noise
        noise_ratio = 0.5; % Noise level
        
        std = 2.55*noise_ratio*rand;
        def_noise = def_float + std * randn(size(def_float));
        ref_noise = ref_float + std * randn(size(ref_float));
    
        %int image
        def_gray = mat2gray(def_float, [0,255]);
        ref_gray = mat2gray(ref_float, [0,255]);
    
        def_gray_noise = mat2gray(def_noise, [0,255]);
        ref_gray_noise = mat2gray(ref_noise, [0,255]);
    
        % save the image pair and ground truth
%         
        directory_image_def = 'path';
        directory_image_ref = 'path'; % The directory where the image pair will be saved
        imwrite(def_gray_noise, [directory_image_def,'deformation',int2str((i-1)*100+j),'.bmp']);
        imwrite(ref_gray_noise, [directory_image_ref,'reference',int2str((i-1)*100+j),'.bmp']);%int_noise
        

        directory_displacement = 'path';
        directory_deformation = 'path'; % The directory where the ground truth will be saved
        save([directory_displacement 'displacement' int2str((i-1)*100+j) '.mat'],'uu')
        save([directory_displacement 'deformation' int2str((i-1)*100+j) '.mat'],'E')
        
    end
end
    
    
%%










