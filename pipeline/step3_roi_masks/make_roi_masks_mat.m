function make_roi_masks_mat()
% Builds roi_masks.mat compatible with SingleTrialDecodingv3.m
% Expects ROI nifti masks like: V1v_in_EPI.nii.gz with shape 53x63x46

%% ===== CONFIG (EDIT THESE) =====
roi_root = '/blue/ruogu.fang/pateld3/neuroimaging/rois_in_epi';      % folder containing ROI nii.gz files
output = '/blue/ruogu.fang/pateld3/neuroimaging/output_mats';
out_path = fullfile(output, 'roi_masks.mat');

% ROI names expected by your decoding script
roi_names = { ...
    'V1v','V1d','V2v','V2d','V3v','V3d','hV4','VO1','VO2', 'IPS' ...
    'PHC1','PHC2','hMT','LO1','LO2','V3a','V3b' ...
};

roi_suffix = '_in_EPI_bin.nii.gz';  % your naming convention
expected_shape = [53 63 46];    % optional safety check; you can infer instead

%% ===== LOAD & BUILD STRUCT =====
roi_masks = struct();

for i = 1:numel(roi_names)
    roi = roi_names{i};
    roi_file = fullfile(roi_root, [roi roi_suffix]);

    if ~isfile(roi_file)
        warning('Missing ROI file: %s (skipping)', roi_file);
        continue;
    end

    % Read nii.gz (gunzip to temp then read)
    [nii_path, tmpFolder] = gunzip_to_temp(roi_file);
    info = niftiinfo(nii_path);
    vol  = niftiread(info);
    cleanup_temp(nii_path, tmpFolder);

    % Sanity check
    if ~isequal(size(vol), expected_shape)
        error('ROI %s has shape [%s], expected [%s]. File: %s', ...
            roi, num2str(size(vol)), num2str(expected_shape), roi_file);
    end

    % Convert to binary mask (nonzero = in ROI)
    mask = vol ~= 0;

    % Store: can store 3D logical or vector; script does (:)>0 so either works
    roi_masks.(roi) = mask;

    fprintf('Loaded ROI %-4s | voxels=%d\n', roi, nnz(mask));
end

%% ===== OPTIONAL: build IPS if you later add IPS0-IPS5 =====
% If you eventually have IPS0_in_EPI.nii.gz ... IPS5_in_EPI.nii.gz
% you can auto-combine them into roi_masks.IPS like your decoding script wants:
%
% ipsNames = {'IPS0','IPS1','IPS2','IPS3','IPS4','IPS5'};
% hasAll = all(isfield(roi_masks, ipsNames));
% if hasAll
%     ipsMask = false(expected_shape);
%     for k=1:numel(ipsNames)
%         ipsMask = ipsMask | roi_masks.(ipsNames{k});
%     end
%     roi_masks.IPS = ipsMask;
% end

%% ===== SAVE =====
save(out_path, '-struct', 'roi_masks'); % saves fields directly (V1v, V1d, ...)
fprintf('\nSaved roi masks to: %s\n', out_path);

end

%% ===== helpers =====
function [nii_path, tmpFolder] = gunzip_to_temp(gz_path)
tmpFolder = tempname;
mkdir(tmpFolder);
gunzip(gz_path, tmpFolder);
files = dir(fullfile(tmpFolder, '*.nii'));
assert(~isempty(files), 'gunzip did not produce .nii for %s', gz_path);
nii_path = fullfile(tmpFolder, files(1).name);
end

function cleanup_temp(nii_path, tmpFolder)
try
    if isfile(nii_path), delete(nii_path); end
    if isfolder(tmpFolder), rmdir(tmpFolder); end
catch
end
end
