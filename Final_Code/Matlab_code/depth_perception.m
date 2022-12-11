clear all 
close all

load('stereoParams.mat')

h1=figure; showReprojectionErrors(stereoParams);

% Visualize pattern locations
h2=figure; showExtrinsics(stereoParams, 'CameraCentric');

%%

I1=imread('Cali_L.jpg');
I2=imread('Cali_R.jpg');

[J1, J2] = rectifyStereoImages(I1, I2, stereoParams);

imshow(stereoAnaglyph(J1,J2));
disparityRange = [48 128+48];
disparityMap = disparitySGM(rgb2gray(J1),rgb2gray(J2),'DisparityRange',disparityRange,'UniquenessThreshold',21);
% disparityMap = disparityBM(rgb2gray(J1),rgb2gray(J2),'DisparityRange',disparityRange,'BlockSize',5,'ContrastThreshold',1,'UniquenessThreshold',7);

figure
imshow(disparityMap,disparityRange)
title('Disparity Map')
colormap jet
colorbar

% pixel from the disparity map.
xyzPoints = reconstructScene(disparityMap, stereoParams);
J=J1;
Z = xyzPoints(:, :, 3);
Z(isnan(Z))=1000000;
mask = repmat(Z<1500   , [1, 1, 3]);
J(~mask) = 0;
figure
imshow(J, 'InitialMagnification', 50);

%%

I1=imread('1m_L.jpg');
I2=imread('1m_R.jpg');

[J1, J2] = rectifyStereoImages(I1, I2, stereoParams);


disparityRange = [48 128+48];
disparityMap = disparitySGM(rgb2gray(J1),rgb2gray(J2),'DisparityRange',disparityRange,'UniquenessThreshold',21);
% disparityMap = disparityBM(rgb2gray(J1),rgb2gray(J2),'DisparityRange',disparityRange,'BlockSize',5,'ContrastThreshold',1,'UniquenessThreshold',7);

% pixel from the disparity map.
xyzPoints = reconstructScene(disparityMap, stereoParams);
J=J1;
Z = xyzPoints(:, :, 3);
Z(isnan(Z))=1000000;
mask = repmat(Z > 1000 & Z<1050   , [1, 1, 3]);
J(~mask) = 0;
figure
imshow(J, 'InitialMagnification', 50);

%%
I1=imread('1.5m_L.jpg');
I2=imread('1.5m_R.jpg');

[J1, J2] = rectifyStereoImages(I1, I2, stereoParams);


disparityRange = [48 128+48];
disparityMap = disparitySGM(rgb2gray(J1),rgb2gray(J2),'DisparityRange',disparityRange,'UniquenessThreshold',21);
% disparityMap = disparityBM(rgb2gray(J1),rgb2gray(J2),'DisparityRange',disparityRange,'BlockSize',5,'ContrastThreshold',1,'UniquenessThreshold',7);


% pixel from the disparity map.
xyzPoints = reconstructScene(disparityMap, stereoParams);
J=J1;
Z = xyzPoints(:, :, 3);
Z(isnan(Z))=1000000;
mask = repmat(Z > 1500 & Z<1600   , [1, 1, 3]);
J(~mask) = 0;
figure
imshow(J, 'InitialMagnification', 50);

%%
I1=imread('2m_L.jpg');
I2=imread('2m_R.jpg');

[J1, J2] = rectifyStereoImages(I1, I2, stereoParams);


disparityRange = [48 128+48];
disparityMap = disparitySGM(rgb2gray(J1),rgb2gray(J2),'DisparityRange',disparityRange,'UniquenessThreshold',21);
% disparityMap = disparityBM(rgb2gray(J1),rgb2gray(J2),'DisparityRange',disparityRange,'BlockSize',5,'ContrastThreshold',1,'UniquenessThreshold',7);


% pixel from the disparity map.
xyzPoints = reconstructScene(disparityMap, stereoParams);
J=J1;
Z = xyzPoints(:, :, 3);
Z(isnan(Z))=1000000;
mask = repmat(Z > 2000 & Z<2100   , [1, 1, 3]);
J(~mask) = 0;
figure
imshow(J, 'InitialMagnification', 50);

%%
I1=imread('2.5m_L.jpg');
I2=imread('2.5m_R.jpg');

[J1, J2] = rectifyStereoImages(I1, I2, stereoParams);


disparityRange = [48 128+48];
disparityMap = disparitySGM(rgb2gray(J1),rgb2gray(J2),'DisparityRange',disparityRange,'UniquenessThreshold',21);
% disparityMap = disparityBM(rgb2gray(J1),rgb2gray(J2),'DisparityRange',disparityRange,'BlockSize',5,'ContrastThreshold',1,'UniquenessThreshold',7);


% pixel from the disparity map.
xyzPoints = reconstructScene(disparityMap, stereoParams);
J=J1;
Z = xyzPoints(:, :, 3);
Z(isnan(Z))=1000000;
mask = repmat(Z > 2500 & Z<2600   , [1, 1, 3]);
J(~mask) = 0;
figure
imshow(J, 'InitialMagnification', 50);
%%
I1=imread('2m_L.jpg');
I2=imread('2m_R.jpg');

[J1, J2] = rectifyStereoImages(I1, I2, stereoParams);


disparityRange = [48 128+48];
disparityMap = disparitySGM(rgb2gray(J1),rgb2gray(J2),'DisparityRange',disparityRange,'UniquenessThreshold',21);
% disparityMap = disparityBM(rgb2gray(J1),rgb2gray(J2),'DisparityRange',disparityRange,'BlockSize',5,'ContrastThreshold',1,'UniquenessThreshold',7);


% pixel from the disparity map.
xyzPoints = reconstructScene(disparityMap, stereoParams);
J=J1;
Z = xyzPoints(:, :, 3);
Z(isnan(Z))=nan;
mask = repmat(Z > 2000 & Z<2100   , [1, 1, 3]);
J(~mask) = 0;
figure
imshow(J1, 'InitialMagnification', 50);
xold = 0;
yold = 0;
k = 0;
hold on;           % and keep it there while we plot
while 1
    [xi, yi, but] = ginput(1);      % get a point
    if ~isequal(but, 1)             % stop if not button 1
        break
    end
    Depth=Z(int16(yi),int16(xi))/1000
    position =  [int16(xi) int16(yi)];
    if isnan(Depth)
        value="Nan";
    else
        value=Depth
    end
    box_color = {'blue'};
    RGB = insertText(J1,position,value,'AnchorPoint','LeftBottom','FontSize',40,'BoxColor',box_color);
    imshow(RGB);
    plot(xi, yi, 'go');         % first point on its own
  

  end