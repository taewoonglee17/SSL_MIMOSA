%%

clear; clc; close all;

set(0,'DefaultFigureWindowStyle','docked')
addpath(genpath('utils'));

%%

savepath = '/autofs/space/marduk_001/users/tommy/mimosa_data/multicoil_train/';
savename = 'mimosa_train.h5';

compare_ref_map     = 1;
load_b1_map         = 1;
b1_type             = 2;

%% LOAD DATA (from .mat instead of DICOM)

fprintf('loading data ... ');
tic

load('/autofs/space/marduk_001/users/tommy/num_phantom_sim_mimosa/numerical_phantom_SSL_input.mat', 'img', 'T1_map', 'T2_map', 'T2std', 'PD_map', 'IE_map','img_b1');

% Name them consistently with previous script
input_img = img;
T2s_map = T2std;

input_img = single(input_img);            
input_img = reshape(input_img, [size(input_img,1), size(input_img,2), size(input_img,3), 1, size(input_img,4)]);
[Nx,Ny,Nz,~,Nacq]  = size(input_img);

IE_map = reshape(IE_map, [size(IE_map,1), size(IE_map,2), 1]);
PD_map = reshape(PD_map, [size(PD_map,1), size(PD_map,2), 1]);
T1_map = reshape(T1_map, [size(T1_map,1), size(T1_map,2), 1]);
T2_map = reshape(T2_map, [size(T2_map,1), size(T2_map,2), 1]);
T2s_map = reshape(T2s_map, [size(T2s_map,1), size(T2s_map,2), 1]);
img_b1 = reshape(img_b1, [size(img_b1,1), size(img_b1,2), 1]);

if load_b1_map == 1
    B1_map = single(img_b1);
    B1_map(B1_map>1.35) = 1.35;
    B1_map(B1_map<0.65) = 0.65;
end

% Make them real valued
input_img = abs(input_img);
T1_map  = abs(single(T1_map))/1000;
T2_map  = abs(single(T2_map))/1000;
T2s_map = abs(single(T2s_map))/1000;
PD_map  = abs(single(PD_map));
IE_map  = abs(single(IE_map));
img_b1 = abs(single(img_b1));

toc
%% Brain intensity statistics

for slc = 1:Nz
    % Root-sum-of-squares over coil/channel dimension (5th dim)
    slc_img = squeeze(rsos(input_img(:,:,slc,1,:),5));

    % Compute statistics
    min_val   = min(slc_img(:));
    max_val   = max(slc_img(:));
    mean_val  = mean(slc_img(:));
    median_val = median(slc_img(:));
    p90_val   = prctile(slc_img(:),90);
    p99_val   = prctile(slc_img(:),99);

    fprintf('Slice %d -> min: %.3f, max: %.3f, mean: %.3f, median: %.3f, p90: %.3f, p99: %.3f\n', ...
            slc, min_val, max_val, mean_val, median_val, p90_val, p99_val);
end
% Brain Mask
threshold = 0.04; % Slice 1 -> min: 0.000, max: 0.102, mean: 0.047, median: 0.082, p90: 0.096, p99: 0.101

[Nx,Ny,Nz,~,~]  = size(input_img);
bmask           = zeros(Nx,Ny,Nz,'single');

for slc = 1:size(input_img,3)
   bmask(:,:,slc) = imfill(squeeze(rsos(input_img(:,:,slc,1,:),5)) > threshold, 'holes');
end

%% Normalize

input_img       = input_img./prctile(abs(input_img(:)),99)./2.5;

sens            = ones(Nx,Ny,Nz,1,'single');
mask            = ones(Nx,Ny,'single');

if compare_ref_map == 0
    T1_map  = ones(Nx,Ny,Nz,'single').*5;
    T2_map  = ones(Nx,Ny,Nz,'single').*2.5;
    T2s_map = ones(Nx,Ny,Nz,'single').*0.05;
    PD_map  = ones(Nx,Ny,Nz,'single');
    IE_map  = ones(Nx,Ny,Nz,'single');
end

if load_b1_map == 0
    B1_map = ones(Nx,Ny,Nz,'single');
end

input_img   = permute(input_img,[2,1,4,3,5]);
sens        = permute(sens,[2,1,4,3]);

T1_map      = permute(T1_map,[2,1,3]);
T2_map      = permute(T2_map,[2,1,3]);
T2s_map     = permute(T2s_map,[2,1,3]);
PD_map      = permute(PD_map,[2,1,3]);
IE_map      = permute(IE_map,[2,1,3]);
B1_map      = permute(B1_map,[2,1,3]);

bmask       = permute(bmask,[2,1,3]);
mask        = permute(mask,[2,1]);

for i = 1:Nacq
    kspace_acq{i} = single(input_img(:,:,:,:,i));
end

%% SAVE DATA

fprintf('save h5 data ... ');
tic

file_name   = strcat(savepath,savename);
att_patient = '0000';
att_seq     = 'QALAS';

if isfile(file_name)
    delete(file_name);
end

for i = 1:Nacq
    h5create(file_name, sprintf('/kspace_acq%d', i), [Ny,Nx,1,1], 'Datatype', 'single');
    h5write(file_name, sprintf('/kspace_acq%d', i), kspace_acq{i});
end

h5create(file_name,'/reconstruction_t1',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_t1', T1_map);
h5create(file_name,'/reconstruction_t2',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_t2', T2_map);
h5create(file_name,'/reconstruction_t2s',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_t2s', T2s_map);
h5create(file_name,'/reconstruction_pd',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_pd', PD_map);
h5create(file_name,'/reconstruction_ie',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_ie', IE_map);
h5create(file_name,'/reconstruction_b1',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_b1', B1_map);

fprintf('mask(:,1) size: [%d, %d]\n', size(mask(1,:)'));
fprintf('Ny: %d\n', Nx);

for i = 1:Nacq
    h5create(file_name,sprintf('/mask_acq%d',i),[Nx,1],'Datatype','single');
    h5write(file_name,sprintf('/mask_acq%d',i), mask(1,:)');
end

% Save masks
h5create(file_name,'/mask_brain',[Ny,Nx,Nz],'Datatype','single');        
h5write(file_name, '/mask_brain', single(bmask));

% Attributes
h5writeatt(file_name,'/','norm_t1',norm(T1_map(:)));
h5writeatt(file_name,'/','max_t1',max(T1_map(:)));
h5writeatt(file_name,'/','norm_t2',norm(T2_map(:)));
h5writeatt(file_name,'/','max_t2',max(T2_map(:)));
h5writeatt(file_name,'/','norm_t2s',norm(T2s_map(:)));
h5writeatt(file_name,'/','max_t2s',max(T2s_map(:)));
h5writeatt(file_name,'/','norm_pd',norm(PD_map(:)));
h5writeatt(file_name,'/','max_pd',max(PD_map(:)));
h5writeatt(file_name,'/','norm_ie',norm(IE_map(:)));
h5writeatt(file_name,'/','max_ie',max(IE_map(:)));
h5writeatt(file_name,'/','norm_b1',norm(B1_map(:)));
h5writeatt(file_name,'/','max_b1',max(B1_map(:)));
h5writeatt(file_name,'/','patient_id',att_patient);
h5writeatt(file_name,'/','acquisition',att_seq);

%% ISMRMRD Header

dset = ismrmrd.Dataset(file_name);

header = [];
header.experimentalConditions.H1resonanceFrequency_Hz   = 128000000;
header.acquisitionSystemInformation.receiverChannels    = 32;
header.encoding.trajectory = 'cartesian';
header.encoding.encodedSpace.fieldOfView_mm.x   = Nx;
header.encoding.encodedSpace.fieldOfView_mm.y   = Ny;
header.encoding.encodedSpace.fieldOfView_mm.z   = Nz;
header.encoding.encodedSpace.matrixSize.x       = Nx*2;
header.encoding.encodedSpace.matrixSize.y       = Ny;
header.encoding.encodedSpace.matrixSize.z       = Nz;
header.encoding.reconSpace.fieldOfView_mm.x     = Nx;
header.encoding.reconSpace.fieldOfView_mm.y     = Ny;
header.encoding.reconSpace.fieldOfView_mm.z     = Nz;
header.encoding.reconSpace.matrixSize.x         = Nx;
header.encoding.reconSpace.matrixSize.y         = Ny;
header.encoding.reconSpace.matrixSize.z         = Nz;
header.encoding.encodingLimits.kspace_encoding_step_1.minimum = 0;
header.encoding.encodingLimits.kspace_encoding_step_1.maximum = Nx-1;
header.encoding.encodingLimits.kspace_encoding_step_1.center  = Nx/2;
header.encoding.encodingLimits.kspace_encoding_step_2.minimum = 0;
header.encoding.encodingLimits.kspace_encoding_step_2.maximum = 0;
header.encoding.encodingLimits.kspace_encoding_step_2.center  = 0;

xmlstring = ismrmrd.xml.serialize(header);
dset.writexml(xmlstring);
dset.close();

toc

copyfile(fullfile(savepath, savename), '/autofs/space/marduk_001/users/tommy/mimosa_data/multicoil_val/mimosa_val.h5');
