%% Print all dataset sizes in an HDF5 (robust to Datatype struct/string)
clear; clc;

% === EDIT THIS PATH IF NEEDED ===
% h5file = '/autofs/space/marduk_001/users/tommy/mimosa_plus_7T_data/multicoil_train/mimosa_plus_7T_train.h5';
h5file = '/autofs/space/marduk_001/users/tommy/mimosa_plus_data/multicoil_train/mimosa_plus_train.h5';

if ~isfile(h5file)
    error('File not found: %s', h5file);
end

info = h5info(h5file);
fprintf('--- HDF5: %s ---\n', h5file);

printGroup(info);  % recursive print

% Count and list kspace acquisitions explicitly from top level (if present)
topNames = {info.Datasets.Name};
isK = startsWith(topNames, 'kspace_acq');
kNames = topNames(isK);
fprintf('\nFound %d top-level k-space acquisitions:\n', numel(kNames));
for i = 1:numel(kNames)
    ds = info.Datasets(strcmp(topNames, kNames{i}));
    fprintf('  /%s -> size %s\n', ds.Name, mat2str(ds.Dataspace.Size));
end


% Top-level attributes
if ~isempty(info.Attributes)
    fprintf('\nTop-level attributes:\n');
    for i = 1:numel(info.Attributes)
        a = info.Attributes(i);
        val = h5readatt(h5file, '/', a.Name);
        if ischar(val)
            vs = val;
        elseif isscalar(val)
            vs = num2str(val);
        else
            vs = sprintf('[%s]', strjoin(string(size(val)), 'x'));
        end
        fprintf('  %-22s = %s\n', a.Name, vs);
    end
end

% ISMRMRD header presence
idxHdr = find(strcmp({info.Datasets.Name}, 'ismrmrd_header'), 1);
if ~isempty(idxHdr)
    ds = info.Datasets(idxHdr);
    fprintf('\nismrmrd_header present -> size %s (XML string bytes)\n', mat2str(ds.Dataspace.Size));
else
    % It might be in a subgroup — try to find it:
    hdrHit = find(strcmp({allDS.name}, 'ismrmrd_header'));
    if ~isempty(hdrHit)
        fprintf('\nismrmrd_header present -> size %s @ %s\n', mat2str(allDS(hdrHit).size), allDS(hdrHit).path);
    else
        fprintf('\nismrmrd_header: (not found)\n');
    end
end

fprintf('\nDone.\n');

%% -------- helper functions (keep below the main script) --------
function printGroup(g)
    % Print datasets in this group
    for d = 1:numel(g.Datasets)
        ds = g.Datasets(d);
        sz = ds.Dataspace.Size;
        dt = localDatatypeToString(ds);
        fprintf('%s/%s  ->  size %s  (Datatype: %s)\n', g.Name, ds.Name, mat2str(sz), dt);
    end
    % Recurse into subgroups
    for gg = 1:numel(g.Groups)
        printGroup(g.Groups(gg));
    end
end

function s = localDatatypeToString(ds)
    % Robustly convert ds.Datatype to a printable string
    s = 'unknown';
    if isfield(ds, 'Datatype')
        dt = ds.Datatype;
        if ischar(dt)
            s = dt;
        elseif isstruct(dt)
            if isfield(dt, 'Class') && ischar(dt.Class)
                s = dt.Class;
            elseif isfield(dt, 'Type') && ischar(dt.Type)
                s = dt.Type;
            else
                s = 'struct';
            end
        end
    end
end

function allDS = collectDatasets(g)
    % Recursively collect all datasets with full paths and sizes
    allDS = struct('path', {}, 'name', {}, 'size', {});
    for d = 1:numel(g.Datasets)
        ds = g.Datasets(d);
        allDS(end+1).path = sprintf('%s/%s', g.Name, ds.Name); %#ok<AGROW>
        allDS(end).name   = ds.Name;
        allDS(end).size   = ds.Dataspace.Size;
    end
    for gg = 1:numel(g.Groups)
        allDS = [allDS, collectDatasets(g.Groups(gg))]; %#ok<AGROW>
    end
end
