clearvars
close all

addpath(genpath(pwd));

I = double(imread('/home/xian/EMOTIC/EmpathyDB_images/ade20k/images/labelme_axxlfpyqbzhgkgf.jpg'));

I = imresize(I,256/min(size(I,1), size(I,2)));

figure
imshow(uint8(I));

% select the method to create the bases
% method = 'edgesandregionsandcorners';
% method = 'edges';
method = 'edgesandregions';
%method =  'superpixels';

% bases parameters
param.minlength = 70; % example: 2; for edges (larger=> less edges)
param.maxdist = 60;%50; % example: 10 in grey scale, 60 in rgb; for segments (larger => less segments)
param.Ncorners = 10; % example: 10; number of corners

% Remove previous bases:
delete bases/*.png

% generate bases --------------------------------------------------------
bases = buildGradientBases(I, method, param);
showBases(bases);

for i = 1:bases.Nbases
    imwrite(bases.B(:,:,:,i), ['bases/base',num2str(i),'.png']);
end






