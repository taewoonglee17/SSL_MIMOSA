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

load('/autofs/space/marduk_001/users/tommy/20250505_mimosa_invivo/mapping_mimosa_R6_yxw_s1_zsssl_dict_v2_b1Cor.mat', 'img_zsssl', 'T1_map', 'T2_map', 'T2s_map', 'PD_map', 'IE_map');
load('/autofs/space/marduk_001/users/tommy/20250505_mimosa_invivo/B1map_yxw_R6_s1.mat', 'img_b1');

input_img = single(img_zsssl);            
input_img = reshape(input_img, [size(input_img,1), size(input_img,2), size(input_img,3), 1, size(input_img,4)]);
[Nx,Ny,Nz,~,Nacq]  = size(input_img);

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

%% Brain Mask (BET-based, using img_zsssl)
mag_multi_echo = abs(img_zsssl(:,:,:,4:end));     % shape: [X, Y, Z, Numer of Input Images]
mag = sqrt(sum(mag_multi_echo.^2, 4));

% Optional: zero out top slices to suppress non-brain signal (can be adjusted or skipped)
mag(1:33,:,:) = 0;

% Run BET
voxel_size  = [1 1 1];
matrix_size = size(mag);
bmask = BET(mag, matrix_size, voxel_size, 0.25, 0);   % 0.6 = fractional intensity threshold

% Ensure single precision (matches original code)
bmask = single(bmask);

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

%% TRIM (X direction only: 60:220)

xrange = 60:220; % X direction crop

% trim all variables in X direction before saving
for i = 1:Nacq
    kspace_acq{i} = kspace_acq{i}(:,xrange,:,:);
end

coil_sens   = sens(:,xrange,:,:);

T1_map      = T1_map(:,xrange,:);
T2_map      = T2_map(:,xrange,:);
T2s_map     = T2s_map(:,xrange,:);
PD_map      = PD_map(:,xrange,:);
IE_map      = IE_map(:,xrange,:);
B1_map      = B1_map(:,xrange,:);

bmask       = bmask(:,xrange,:);
mask        = mask(:,xrange);

%% ROTATE ALL IMAGES 90 DEGREES COUNTERCLOCKWISE, THEN FLIP UPSIDE DOWN

fprintf('rotating and flipping image volumes ... ');
tic

% Helper function to rotate 3D volume
% rotate_volume = @(vol) permute(rot90(permute(vol, [2,1,3])), [2,1,3]);

% Rotate and flip the image maps
T1_map  = fliplr(rot90(T1_map));
T2_map  = fliplr(rot90(T2_map));
T2s_map = fliplr(rot90(T2s_map));
PD_map  = fliplr(rot90(PD_map));
IE_map  = fliplr(rot90(IE_map));
B1_map  = fliplr(rot90(B1_map));
bmask   = fliplr(rot90(bmask));     % brain mask
mask   = fliplr(rot90(mask)); 

% Rotate and flip k-space data for each acquisition
for i = 1:Nacq
    kspace_acq{i} = fliplr(rot90(kspace_acq{i}));
end

% Update dimensions
[Nx, Ny, ~] = size(T1_map);  % these are now updated after rotation + flip

toc


%% SAVE DATA

fprintf('save h5 data ... ');
tic

file_name   = strcat(savepath,savename);
att_patient = '0000';
att_seq     = 'QALAS';

for i = 1:Nacq
    kspace_acq{i} = permute(kspace_acq{i},[4,3,2,1]); % flips when making h5 file, so we unflip
end
coil_sens = permute(coil_sens,[4,3,2,1]);

data_to_save = struct();
for i = 1:Nacq
    field_name = sprintf('kspace_acq%d', i);
    data_to_save.(field_name) = kspace_acq{i};
end

% % If complex-valued:
% saveh5(data_to_save, file_name, 'ComplexFormat', {'r','i'}, 'RootName', '/');
% If real-valued:
saveh5(data_to_save, file_name, 'RootName', '/');

h5create(file_name,'/reconstruction_t1',[Nx,Ny,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_t1', T1_map);
h5create(file_name,'/reconstruction_t2',[Nx,Ny,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_t2', T2_map);
h5create(file_name,'/reconstruction_t2s',[Nx,Ny,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_t2s', T2s_map);
h5create(file_name,'/reconstruction_pd',[Nx,Ny,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_pd', PD_map);
h5create(file_name,'/reconstruction_ie',[Nx,Ny,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_ie', IE_map);
h5create(file_name,'/reconstruction_b1',[Nx,Ny,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_b1', B1_map);

fprintf('mask(:,1) size: [%d, %d]\n', size(mask(1,:)'));
fprintf('Ny: %d\n', Ny);

for i = 1:Nacq
    h5create(file_name,sprintf('/mask_acq%d',i),[Ny,1],'Datatype','single');
    h5write(file_name,sprintf('/mask_acq%d',i), mask(1,:)');
end

h5create(file_name,'/mask_brain',[Nx,Ny,Nz],'Datatype','single');
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