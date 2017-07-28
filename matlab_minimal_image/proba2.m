clearvars
close all

addpath(genpath(pwd));

class_names = {'tallbuilding', 'mountain', 'Opencountry', 'street', ...
    'inside_city', 'highway', 'forest', 'coast'};

% I = double(imread('images/art608.jpg')); truelabel = 6;
% I = double(imread('images/art392.jpg')); truelabel = 1;
% I = double(imread('images/art764.jpg')); truelabel = 4;
% I = double(imread('images/bea14.jpg')); truelabel = 8;
% I = double(imread('images/for157.jpg')); truelabel = 7;
% I = double(imread('images/land18.jpg')); truelabel = 2;
% I = double(imread('images/land964.jpg')); truelabel = 3;
I = double(imread('images/urb307.jpg')); truelabel = 5;

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



% generate bases --------------------------------------------------------
bases = buildGradientBases(I, method, param);
% showBases(bases);
