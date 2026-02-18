function extract_betas()
% Extract single-trial betas from SPM output (BetaS2.m) into
% Pl#.mat, Nt#.mat, Up#.mat for SingleTrialDecodingv3.m
%
% BetaS2.m creates one SPM model per subject with 5 sessions.
% Each session has 60 condition regressors + 6 motion regressors = 66.
% Condition order per session:
%   1-20  : Pl1..Pl20  (Pleasant)
%   21-40 : Nt1..Nt20  (Neutral)
%   41-60 : Up1..Up20  (Unpleasant)
%   61-66 : motion (X,Y,Z,x,y,z)
%
% SPM beta ordering across sessions:
%   Session 1: beta_0001..beta_0066
%   Session 2: beta_0067..beta_0132
%   ...
% followed by session constants at the end.

%% ===== CONFIG =====
beta_root = '/orange/ruogu.fang/pateld3/SPM_Preprocessed_fMRI_20Subjects/betas';
out_dir   = '/blue/ruogu.fang/pateld3/neuroimaging/output_mats';

num_subjects = 20;
num_runs     = 5;
conds_per_run = 60;   % 20 Pl + 20 Nt + 20 Up
motion_per_run = 6;
regs_per_run = conds_per_run + motion_per_run;  % 66

mkdir(out_dir);

%% ===== MAIN =====
for s = 1:num_subjects
    fprintf('\n=== Subject %d ===\n', s);

    beta_dir = fullfile(beta_root, sprintf('beta_series_%d', s));
    if ~isfolder(beta_dir)
        warning('Missing beta dir: %s. Skipping subject %d.', beta_dir, s);
        continue;
    end

    Pl = [];
    Nt = [];
    Up = [];

    for r = 1:num_runs
        offset = (r - 1) * regs_per_run;

        pl_indices = offset + (1:20);
        nt_indices = offset + (21:40);
        up_indices = offset + (41:60);

        Pl_run = load_beta_images(beta_dir, pl_indices);
        Nt_run = load_beta_images(beta_dir, nt_indices);
        Up_run = load_beta_images(beta_dir, up_indices);

        Pl = [Pl, Pl_run];
        Nt = [Nt, Nt_run];
        Up = [Up, Up_run];

        fprintf('  Run %d: Pl=%d  Nt=%d  Up=%d\n', r, size(Pl_run,2), size(Nt_run,2), size(Up_run,2));
    end

    fprintf('  Totals: Pl=%d  Nt=%d  Up=%d  voxels=%d\n', ...
        size(Pl,2), size(Nt,2), size(Up,2), size(Pl,1));

    save(fullfile(out_dir, sprintf('Pl%d.mat', s)), 'Pl', '-v7.3');
    save(fullfile(out_dir, sprintf('Nt%d.mat', s)), 'Nt', '-v7.3');
    save(fullfile(out_dir, sprintf('Up%d.mat', s)), 'Up', '-v7.3');

    fprintf('  Saved Pl%d.mat Nt%d.mat Up%d.mat\n', s, s, s);
end
end

%% ===== Helper =====
function X = load_beta_images(beta_dir, indices)
% Load SPM beta images and return [nVox x nTrials] matrix
nT = numel(indices);
X  = [];

for k = 1:nT
    beta_file = fullfile(beta_dir, sprintf('beta_%04d.nii', indices(k)));

    % SPM may also write .img/.hdr pairs
    if ~isfile(beta_file)
        beta_file_img = fullfile(beta_dir, sprintf('beta_%04d.img', indices(k)));
        if isfile(beta_file_img)
            beta_file = beta_file_img;
        else
            error('Missing beta file: %s (.nii or .img)', beta_file);
        end
    end

    vol = spm_read_vols(spm_vol(beta_file));
    vec = single(vol(:));

    if isempty(X)
        X = zeros(numel(vec), nT, 'single');
    end
    X(:, k) = vec;
end
end
