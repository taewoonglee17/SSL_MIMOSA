function mimosa_slice_viewer_and_export()
% Minimal slice viewer + PNG exporter with colormap and intensity range control.
% - Colormap: grayscale or hot
% - Intensity ranges:
%     * Absolute: Imin/Imax (used if both are valid and Imin < Imax)
%     * Else fallback: Percentiles Pmin/Pmax (default 1–99)

clc; close all;

%% -------- Load data (your paths) --------
fprintf('Loading data ... ');
tic
load('/autofs/space/marduk_001/users/tommy/20241001_invivo/img_R11_zsssl_yohan_tf2_un8cg8res5_ch256_reorder.mat', 'img_zsssl_reorder');
load('/autofs/space/marduk_001/users/tommy/20241001_invivo/mapping_allslc_zsssl_tfv4_dict_v2.mat', ...
     'T1_map', 'T2_map', 'T2std', 'PD_map', 'IE_map');
T2s_map = T2std; clear T2std;
load('/autofs/space/marduk_001/users/tommy/20241001_invivo/img_b1.mat', 'img_b1');
load('/autofs/space/marduk_001/users/tommy/20241001_invivo/mask_brain.mat','mask');
load('/autofs/space/marduk_001/users/tommy/20241001_invivo/T1_IR_reg_linear.mat','T1_IR_reg');
load('/autofs/space/marduk_001/users/tommy/20241001_invivo/T2_SE_reg_linear.mat','T2_SE_reg');
load('/autofs/space/marduk_001/users/tommy/20241001_invivo/T2std_20240921_rgs_to_20241001_slc153.mat','T2std');
toc

% New target maps (global reference)
vols.T1_global  = abs(single(T1_IR_reg));                       %T1 map
vols.T2_global  = abs(single(T2_SE_reg));                       % T2 map
vols.T2s_global = abs(single(T2std));                           % T2* map
clear T2std;

vols.im_cs_pad = abs(single(img_zsssl_reorder));   % 4D [Y X Z A]
vols.T1_map    = abs(single(T1_map));
vols.T2_map    = abs(single(T2_map));
vols.T2s_map   = abs(single(T2s_map));
vols.PD_map    = abs(single(PD_map));
vols.IE_map    = abs(single(IE_map));
vols.img_b1    = abs(single(img_b1));
% Try to load a real mask; fallback to zeros if missing
vols.bmask     = abs(single(mask));

names = {'im_cs_pad','T1_map','T2_map','T2s_map','PD_map','IE_map','img_b1','bmask','T1_global','T2_global','T2s_global'};

%% -------- Dimension report --------
fprintf('\n=== DIMENSIONS ===\n');
for k = 1:numel(names)
    nm = names{k};
    if isfield(vols,nm)
        sz = size(vols.(nm));
        shapeStr = sprintf('%dx', sz); shapeStr = shapeStr(1:end-1);
        fprintf('%-10s : %s (ndims=%d)\n', nm, shapeStr, ndims(vols.(nm)));
    end
end
fprintf('===================\n\n');

%% -------- Figure & Controls (classic uicontrols) --------
f = figure('Name','MIMOSA Slice Viewer + Export', 'NumberTitle','off', ...
           'MenuBar','none','ToolBar','none', 'Color','k', ...
           'Units','normalized','Position',[0.2 0.1 0.6 0.80]);

ax = axes('Parent',f,'Position',[0.08 0.24 0.84 0.72]); %#ok<LAXES>
axis(ax,'image'); ax.YDir = 'normal'; ax.Visible = 'off';
colormap(ax, gray(256));

% Dataset
uicontrol(f,'Style','text','String','Dataset','Units','normalized',...
    'Position',[0.08 0.175 0.10 0.03],'BackgroundColor','k','ForegroundColor','w','FontWeight','bold','HorizontalAlignment','left');
ddData = uicontrol(f,'Style','popupmenu','String',names,'Value',1, ...
    'Units','normalized','Position',[0.18 0.18 0.18 0.035], 'Callback',@onDataset);

% View
uicontrol(f,'Style','text','String','View','Units','normalized',...
    'Position',[0.38 0.175 0.06 0.03],'BackgroundColor','k','ForegroundColor','w','FontWeight','bold','HorizontalAlignment','left');
ddView = uicontrol(f,'Style','popupmenu','String',{'Axial (XY@Z)','Sagittal (XZ@Y)','Coronal (YZ@X)'},'Value',1, ...
    'Units','normalized','Position',[0.44 0.18 0.18 0.035], 'Callback',@(~,~) redraw());

% Acquisition (only for 4D im_cs_pad)
txtAcq = uicontrol(f,'Style','text','String','Acq','Units','normalized',...
    'Position',[0.64 0.175 0.05 0.03],'BackgroundColor','k','ForegroundColor','w','FontWeight','bold','HorizontalAlignment','left');
slAcq = uicontrol(f,'Style','slider','Units','normalized','Position',[0.69 0.185 0.23 0.02], ...
    'Min',1,'Max',max(2,size(vols.im_cs_pad,4)),'Value',1,'SliderStep',[1 1]./max(1,(size(vols.im_cs_pad,4)-1)), ...
    'Callback',@(~,~) redraw());

% X/Y/Z sliders
mkText = @(lbl,y) uicontrol(f,'Style','text','String',lbl,'Units','normalized',...
    'Position',[0.08 y 0.02 0.03],'BackgroundColor','k','ForegroundColor','w','FontWeight','bold','HorizontalAlignment','left');
mkSld  = @(y) uicontrol(f,'Style','slider','Units','normalized','Position',[0.12 y+0.01 0.80 0.02], ...
    'Min',1,'Max',2,'Value',1,'SliderStep',[1 1]./1,'Callback',@(~,~) redraw());

mkText('X',0.135); slX = mkSld(0.135);
mkText('Y',0.105); slY = mkSld(0.105);
mkText('Z',0.075); slZ = mkSld(0.075);

% ---- NEW: Colormap + Ranges ----
uicontrol(f,'Style','text','String','Colormap','Units','normalized',...
    'Position',[0.08 0.045 0.09 0.03],'BackgroundColor','k','ForegroundColor','w','FontWeight','bold','HorizontalAlignment','left');
ddCmap = uicontrol(f,'Style','popupmenu','String',{'grayscale','hot'},'Value',1, ...
    'Units','normalized','Position',[0.17 0.05 0.12 0.035], 'Callback',@(~,~) redraw());

% Absolute intensity range
uicontrol(f,'Style','text','String','Imin','Units','normalized',...
    'Position',[0.31 0.045 0.05 0.03],'BackgroundColor','k','ForegroundColor','w','FontWeight','bold','HorizontalAlignment','left');
edImin = uicontrol(f,'Style','edit','String','','Units','normalized','Position',[0.35 0.05 0.08 0.035], ...
                   'Callback',@(~,~) redraw());
uicontrol(f,'Style','text','String','Imax','Units','normalized',...
    'Position',[0.45 0.045 0.05 0.03],'BackgroundColor','k','ForegroundColor','w','FontWeight','bold','HorizontalAlignment','left');
edImax = uicontrol(f,'Style','edit','String','','Units','normalized','Position',[0.49 0.05 0.08 0.035], ...
                   'Callback',@(~,~) redraw());

% Percentile fallback (used if Imin/Imax are not both valid)
uicontrol(f,'Style','text','String','Pmin','Units','normalized',...
    'Position',[0.59 0.045 0.05 0.03],'BackgroundColor','k','ForegroundColor','w','FontWeight','bold','HorizontalAlignment','left');
edPmin = uicontrol(f,'Style','edit','String','1','Units','normalized','Position',[0.63 0.05 0.06 0.035], ...
                   'Callback',@(~,~) redraw());
uicontrol(f,'Style','text','String','Pmax','Units','normalized',...
    'Position',[0.71 0.045 0.05 0.03],'BackgroundColor','k','ForegroundColor','w','FontWeight','bold','HorizontalAlignment','left');
edPmax = uicontrol(f,'Style','edit','String','99','Units','normalized','Position',[0.75 0.05 0.06 0.035], ...
                   'Callback',@(~,~) redraw());

% Save PNG
btnSave = uicontrol(f,'Style','pushbutton','String','Save PNG','Units','normalized',...
    'Position',[0.83 0.045 0.13 0.04],'Callback',@onSave);

% State
S.name = names{1};
S.outdir = '';   % set on first save
[V, is4D] = getVol(vols, S.name, 1);
[Ny,Nx,Nz] = size3(V);
setXYZRanges(Nx,Ny,Nz);
setAcqVisibility(is4D, size4(vols, S.name));

% Initial draw
redraw();

%% --------- Callbacks ----------
    function onDataset(~,~)
        S.name = names{get(ddData,'Value')};
        if strcmp(S.name,'im_cs_pad'), set(slAcq,'Value',1); end
        [V, is4D] = getVol(vols, S.name, round(get(slAcq,'Value')));
        [Ny,Nx,Nz] = size3(V);
        setXYZRanges(Nx,Ny,Nz);
        setAcqVisibility(is4D, size4(vols, S.name));
        redraw();
    end

    function onSave(~,~)
        % Get current 2D image and windowing/colormap
        [img, lo, hi] = currentImageAndWindow();
        cmapName = currentCmap();
        % Ask output folder if not set
        if isempty(S.outdir) || ~isfolder(S.outdir)
            S.outdir = uigetdir(pwd,'Select output directory for PNG slices');
            if isequal(S.outdir,0), return; end
        end
        % Compose filename
        acq = round(get(slAcq,'Value'));
        x = round(get(slX,'Value')); y = round(get(slY,'Value')); z = round(get(slZ,'Value'));
        base = sprintf('%s_%s_x%d_y%d_z%d', S.name, upper(viewName()), x, y, z);
        if strcmp(S.name,'im_cs_pad'), base = sprintf('%s_acq%d', base, acq); end
        base = sprintf('%s_%s', base, cmapName); % include colormap
        fname = fullfile(S.outdir, [base '.png']);

        % Save according to colormap
        if strcmp(cmapName,'grayscale')
            lims = double([lo, hi]);
            img8 = uint8(255 * mat2gray(double(img), lims));
            imwrite(img8, fname);
        else
            % hot colormap → truecolor RGB
            ctab = hot(256);
            img01 = (double(img) - double(lo)) / max(double(hi - lo), eps);
            img01 = min(max(img01, 0), 1);
            idx = max(1, min(256, round(img01 * 255) + 1));
            rgb = ind2rgb(idx, ctab);
            imwrite(rgb, fname);
        end
        fprintf('Saved: %s\n', fname);
    end

%% --------- Utility ----------
    function setXYZRanges(nx,ny,nz)
        set(slX,'Min',1,'Max',max(2,nx),'Value',ceil(nx/2),'SliderStep',[1 1]./max(1,(nx-1)));
        set(slY,'Min',1,'Max',max(2,ny),'Value',ceil(ny/2),'SliderStep',[1 1]./max(1,(ny-1)));
        set(slZ,'Min',1,'Max',max(2,nz),'Value',ceil(nz/2),'SliderStep',[1 1]./max(1,(nz-1)));
    end

    function setAcqVisibility(is4, na)
        if is4
            set([txtAcq,slAcq],'Visible','on');
            set(slAcq,'Min',1,'Max',max(2,na),'SliderStep',[1 1]./max(1,(na-1)));
        else
            set([txtAcq,slAcq],'Visible','off');
        end
    end

    function [V,is4] = getVol(dict, key, acq)
        A = dict.(key);
        if ndims(A) == 4
            is4 = true;
            na  = size(A,4);
            acq = max(1, min(na, acq));
            V   = A(:,:,:,acq);
        else
            is4 = false;
            V   = A;
        end
        if ~isa(V,'single') && ~isa(V,'double'), V = single(V); end
    end

    function n = size4(dict, key)
        A = dict.(key);
        if ndims(A) == 4, n = size(A,4); else, n = 1; end
    end

    function [ny,nx,nz] = size3(A)
        s = size(A);
        if numel(s)<3, s(3)=1; end
        ny = s(1); nx = s(2); nz = s(3);
    end

    function cmapName = currentCmap()
        cmapName = get(ddCmap,'String');
        cmapName = cmapName{get(ddCmap,'Value')};
    end

    function [useAbs, imin, imax] = readAbsRange()
        % Return whether a valid absolute range is present
        imin = str2double(get(edImin,'String'));
        imax = str2double(get(edImax,'String'));
        useAbs = isfinite(imin) && isfinite(imax) && (imax > imin);
    end

    function [p1,p2] = readPercentiles()
        p1 = str2double(get(edPmin,'String')); if isnan(p1), p1 = 1; end
        p2 = str2double(get(edPmax,'String')); if isnan(p2), p2 = 99; end
        p1 = max(0, min(100, p1));
        p2 = max(0, min(100, p2));
        if p2 <= p1, p2 = p1 + eps; end
    end

    function [img, lo, hi] = currentImageAndWindow()
        % Grab current 2D slice + compute window [lo, hi]
        acq = round(get(slAcq,'Value'));
        [V, ~] = getVol(vols, S.name, acq);
        viewMode = get(ddView,'Value'); % 1=axial, 2=sagittal, 3=coronal
        x = max(1, min(round(get(slX,'Value')), size(V,2)));
        y = max(1, min(round(get(slY,'Value')), size(V,1)));
        z = max(1, min(round(get(slZ,'Value')), size(V,3)));
        switch viewMode
            case 1, img = squeeze(V(:,:,z));
            case 2, img = squeeze(V(y,:,:))';
            case 3, img = squeeze(V(:,x,:))';
        end

        [useAbs, imin, imax] = readAbsRange();
        if useAbs
            lo = imin; hi = imax;
        else
            [p1,p2] = readPercentiles();
            lo = prctile(img(:), p1);
            hi = prctile(img(:), p2);
            if hi <= lo, hi = lo + eps; end
        end
    end

    function s = viewName()
        switch get(ddView,'Value')
            case 1, s = 'axial';
            case 2, s = 'sagittal';
            case 3, s = 'coronal';
        end
    end

%% --------- Draw ---------
    function redraw()
        [img, lo, hi] = currentImageAndWindow();
        imagesc(ax, img, [lo hi]); axis(ax,'image'); ax.YDir='normal';

        % Apply chosen colormap
        switch currentCmap()
            case 'grayscale', colormap(ax, gray(256));
            case 'hot',       colormap(ax, hot(256));
        end

        % ---- NEW: coordinate overlay text ----
        x = round(get(slX,'Value'));
        y = round(get(slY,'Value'));
        z = round(get(slZ,'Value'));
        acq = round(get(slAcq,'Value'));
        coordsText = sprintf('X=%d | Y=%d | Z=%d', x, y, z);
        if strcmp(S.name,'im_cs_pad')
            coordsText = sprintf('%s | Acq=%d', coordsText, acq);
        end
        % Display in lower-right corner (white text with black edge)
        text(ax, size(img,2)-10, size(img,1)-10, coordsText, ...
            'Color','w','FontSize',10,'FontWeight','bold', ...
            'HorizontalAlignment','right','VerticalAlignment','bottom', ...
            'BackgroundColor','k','Margin',2,'Interpreter','none');

        % ---- Title ----
        ttl = sprintf('%s | %s%s', S.name, upper(viewName()), ...
            strcmp(S.name,'im_cs_pad') * sprintf(' | acq=%d', acq));
        title(ax, ttl, 'Color','w','FontWeight','bold');

        ax.Visible = 'on';
        drawnow;
    end

end
