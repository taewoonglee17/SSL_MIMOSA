%% Full-volume version (48 slices) with original pipeline kept intact
% - Keeps IE dictionary from toolbox (saved to /reconstruction_ie)
% - Keeps IE regression map (saved to /reconstruction_ie_regression)
% - Keeps original save paths, attributes, and ISMRMRD header format
% - Only changes: data loading (48-slice .mat) + brain masking method

clear; clc; close all;
set(0,'DefaultFigureWindowStyle','docked')
addpath(genpath('utils'));

%% I/O and options (unchanged paths/names)
savepath = '/autofs/space/marduk_001/users/tommy/mimosa_plus_data/multicoil_train/';
savename = 'mimosa_plus_train.h5';
iepath   = '/autofs/cluster/berkin/berkin/Matlab_Code_New/qalas_7T/Qalas_Inversion_Efficiency_Toolbox_v4/utils/inversion-efficiency-simulation/data/';

% New phantom .mat (all 48 slices)
mat_path = '/autofs/space/marduk_001/users/tommy/20240820_nist/maps_allslc_IElkp_dict_doub.mat';

compare_ref_map     = 1;
load_b1_map         = 1;
b1_type             = 2; %#ok<NASGU> % kept for completeness

% Per-slice mask threshold (fraction of each slice's 99th percentile)
zero_threshold_frac = 0.05;   % 0.5%; increase to shrink mask, decrease to expand

%% LOAD DATA (from .mat) — full volume
fprintf('loading data ... ');
tic
load(mat_path, 'img','T1_map','T2_map','T2std','PD_map','IE_map','img_b1');

% Keep names consistent with original script
T2s_map  = T2std;  clear T2std;
input_img = img;   clear img;

% Load IE Dictionary exactly like original
load([iepath,'ielookup_toolbox_v3.mat']);
IE_dict = single(cat(2, ielookup.ies_mtx, ...
                     cat(1, ielookup.T1', zeros(length(ielookup.T2)-length(ielookup.T1),1)), ...
                     ielookup.T2));

input_img = single(input_img);            
input_img = reshape(input_img, [size(input_img,1), size(input_img,2), size(input_img,3), 1, size(input_img,4)]);
[Nx,Ny,Nz,~,Nacq]  = size(input_img);

% Parameter maps (keep all slices; no collapsing)
IE_map  = single(IE_map);
PD_map  = single(PD_map);
T1_map  = single(T1_map);
T2_map  = single(T2_map);
T2s_map = single(T2s_map);
img_b1  = single(img_b1);

% Optional B1 clamping (same as original)
if load_b1_map == 1
    B1_map = single(img_b1);
    B1_map(B1_map>1.35) = 1.35;
    B1_map(B1_map<0.65) = 0.65;
end

% Real-valued & units (ms -> s for T1/T2/T2s)
input_img = abs(input_img);
T1_map  = abs(T1_map)/1000;
T2_map  = abs(T2_map)/1000;
T2s_map = abs(T2s_map)/1000;
PD_map  = abs(PD_map);
IE_map  = abs(IE_map);
img_b1  = abs(img_b1);
toc

%% Brain masks
% (1) bmask_rmse: keep the old ROI-sum approach (2D union replicated over Nz)
roiData = load('/autofs/space/marduk_001/users/tommy/20240820_nist/mask_rois_T2pla.mat', 'mask_rois');
bmask_rmse_2d = sum(single(roiData.mask_rois), 3) > 0;   % 2-D ROI union
bmask_rmse    = repmat(bmask_rmse_2d, [1, 1, Nz]);

% (2) bmask: per-slice SoS thresholding across acquisitions
sos = squeeze(sqrt(sum(abs(input_img).^2, 5)));  % [Nx, Ny, Nz]
bmask = zeros(Nx,Ny,Nz,'single');
for z = 1:Nz
    slice = sos(:,:,z);
    p99   = prctile(slice(:), 99);
    thr   = max(1e-6, zero_threshold_frac * p99);
    bmask(:,:,z) = single(slice > thr);
end

%% Normalize (same as original)
input_img = input_img ./ prctile(abs(input_img(:)),99) ./ 2.5;

% Dummy sens/mask (unchanged)
sens = ones(Nx,Ny,Nz,1,'single');
mask = ones(Nx,Ny,'single');

% Optional: overwrite maps with constants (unchanged behavior)
if compare_ref_map == 0
    T1_map  = ones(Nx,Ny,Nz,'single').*5;
    T2_map  = ones(Nx,Ny,Nz,'single').*2.5;
    T2s_map = ones(Nx,Ny,Nz,'single').*0.05;
    PD_map  = ones(Nx,Ny,Nz,'single');
    IE_map  = ones(Nx,Ny,Nz,'single');
end

%% Permute to match prior H5 layout ([Ny, Nx, ...])
input_img   = permute(input_img,[2,1,4,3,5]);   % -> [Ny, Nx, 1, Nz, Nacq]
sens        = permute(sens,[2,1,4,3]);          % -> [Ny, Nx, 1, Nz]

T1_map      = permute(T1_map,[2,1,3]);          % -> [Ny, Nx, Nz]
T2_map      = permute(T2_map,[2,1,3]);
T2s_map     = permute(T2s_map,[2,1,3]);
PD_map      = permute(PD_map,[2,1,3]);
IE_map      = permute(IE_map,[2,1,3]);
IE_dict     = permute(IE_dict,[2,1,3]);         % keep exactly like original
B1_map      = permute(B1_map,[2,1,3]);

bmask       = permute(bmask,[2,1,3]);           % -> [Ny, Nx, Nz]
bmask_rmse  = permute(bmask_rmse,[2,1,3]);      % -> [Ny, Nx, Nz]
mask        = permute(mask,[2,1]);              % -> [Ny, Nx]

% Pack acquisitions
kspace_acq = cell(1, Nacq);
for i = 1:Nacq
    kspace_acq{i} = single(input_img(:,:,:,:,i));   % [Ny, Nx, 1, Nz]
end

%% SAVE DATA (same paths and attributes as original)
fprintf('save h5 data ... ');
tic
file_name   = strcat(savepath,savename);
att_patient = '0000';
att_seq     = 'QALAS';

if isfile(file_name)
    delete(file_name);
end

% k-space per acquisition (now includes all Nz slices)
for i = 1:Nacq
    h5create(file_name, sprintf('/kspace_acq%d', i), [Ny,Nx,1,Nz], 'Datatype', 'single');
    h5write(file_name,  sprintf('/kspace_acq%d', i), kspace_acq{i});
end

% Recon maps (keep IE_dict and IE_map exactly like original)
h5create(file_name,'/reconstruction_t1', [Ny,Nx,Nz],'Datatype','single');   h5write(file_name,'/reconstruction_t1', T1_map);
h5create(file_name,'/reconstruction_t2', [Ny,Nx,Nz],'Datatype','single');   h5write(file_name,'/reconstruction_t2', T2_map);
h5create(file_name,'/reconstruction_t2s',[Ny,Nx,Nz],'Datatype','single');   h5write(file_name,'/reconstruction_t2s',T2s_map);
h5create(file_name,'/reconstruction_pd', [Ny,Nx,Nz],'Datatype','single');   h5write(file_name,'/reconstruction_pd', PD_map);

% IE dictionary (toolbox lookup) — unchanged from original
h5create(file_name,'/reconstruction_ie', [size(IE_dict,1),size(IE_dict,2),size(IE_dict,3)],'Datatype','single');
h5write(file_name, '/reconstruction_ie', IE_dict);

% IE regression map — unchanged from original
h5create(file_name,'/reconstruction_ie_regression',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_ie_regression', IE_map);

% B1
h5create(file_name,'/reconstruction_b1',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_b1', B1_map);

% 1-D mask for each acquisition (unchanged)
fprintf('mask(:,1) size: [%d, %d]\n', size(mask(1,:)'));
fprintf('Ny: %d\n', Nx);
for i = 1:Nacq
    h5create(file_name,sprintf('/mask_acq%d',i),[Nx,1],'Datatype','single');
    h5write(file_name,sprintf('/mask_acq%d',i), single(mask(1,:)'));
end

% Save masks
h5create(file_name,'/mask_brain',[Ny,Nx,Nz],'Datatype','single');          % new per-slice SoS mask
h5write(file_name, '/mask_brain', single(bmask));
h5create(file_name,'/mask_brain_rmse',[Ny,Nx,Nz],'Datatype','single');     % original ROI-union replicated
h5write(file_name, '/mask_brain_rmse', single(bmask_rmse));

% Attributes (same keys as original)
h5writeatt(file_name,'/','norm_t1',norm(T1_map(:)));
h5writeatt(file_name,'/','max_t1',max(T1_map(:)));
h5writeatt(file_name,'/','norm_t2',norm(T2_map(:)));
h5writeatt(file_name,'/','max_t2',max(T2_map(:)));
h5writeatt(file_name,'/','norm_t2s',norm(T2s_map(:)));
h5writeatt(file_name,'/','max_t2s',max(T2s_map(:)));
h5writeatt(file_name,'/','norm_pd',norm(PD_map(:)));
h5writeatt(file_name,'/','max_pd',max(PD_map(:)));
h5writeatt(file_name,'/','norm_ie',norm(IE_dict(:)));              % dictionary norm like original
h5writeatt(file_name,'/','max_ie',max(IE_dict(:)));
h5writeatt(file_name,'/','norm_ie_regression',norm(IE_map(:)));    % also include regression map norm
h5writeatt(file_name,'/','max_ie_regression',max(IE_map(:)));
h5writeatt(file_name,'/','norm_b1',norm(B1_map(:)));
h5writeatt(file_name,'/','max_b1',max(B1_map(:)));
h5writeatt(file_name,'/','patient_id',att_patient);
h5writeatt(file_name,'/','acquisition',att_seq);

%% ISMRMRD Header (unchanged)
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

% Also copy to validation split (keep original path/name)
copyfile(fullfile(savepath, savename), '/autofs/space/marduk_001/users/tommy/mimosa_plus_data/multicoil_val/mimosa_plus_val.h5');
