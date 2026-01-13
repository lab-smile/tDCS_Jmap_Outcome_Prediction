% registration_all_auto_part_Jbrain_components.m
%
% SPM12 Normalize: Estimate & Write
% - Estimate deformation from T1.nii
% - Apply the same deformation to:
%     T1_tDCSLAB_Jbrain.nii
%     T1_tDCSLAB_Jbrain_x_fromEmag.nii
%     T1_tDCSLAB_Jbrain_y_fromEmag.nii
%     T1_tDCSLAB_Jbrain_z_fromEmag.nii
%
% Output prefix: w
% Produces (per subject):
%   wT1_tDCSLAB_Jbrain.nii
%   wT1_tDCSLAB_Jbrain_x_fromEmag.nii
%   wT1_tDCSLAB_Jbrain_y_fromEmag.nii
%   wT1_tDCSLAB_Jbrain_z_fromEmag.nii

clear; clc;

rootDir = '/orange/ruogu.fang/junfu.cheng/SMILE/j_map/j_map_direction/read_matlab_from_skylar';
folders = dir(rootDir);

% Files to normalize (resample) using deformation estimated from T1.nii
resample_files = { ...
    'T1_tDCSLAB_Jbrain.nii', ...
    'T1_tDCSLAB_Jbrain_x_fromEmag.nii', ...
    'T1_tDCSLAB_Jbrain_y_fromEmag.nii', ...
    'T1_tDCSLAB_Jbrain_z_fromEmag.nii' ...
};

% Parameters to control which chunk to process
numParts  = 4;   % Total number of parts to split the job
partToRun = 3;   % Change this to select which part to run (1-based index)

%% ---- FIND NUMERIC SUBJECT FOLDERS ----
folderList = folders([folders.isdir] & ...
    ~ismember({folders.name}, {'.', '..'}) & ...
    ~cellfun(@isempty, regexp({folders.name}, '^\d+$', 'once')));

% Sort folder names numerically
folderNames = {folderList.name};
[~, sortIdx] = sort(str2double(folderNames));
folderList = folderList(sortIdx);

% Compute partition indices
totalFolders   = length(folderList);
foldersPerPart = ceil(totalFolders / numParts);
startIdx       = (partToRun - 1) * foldersPerPart + 1;
endIdx         = min(partToRun * foldersPerPart, totalFolders);

fprintf('Total folders: %d | Running part %d/%d => idx %d to %d\n', ...
    totalFolders, partToRun, numParts, startIdx, endIdx);

%% ---- LOOP SUBJECTS ----
for i = startIdx:endIdx
    folderName = folderList(i).name;
    folderPath = fullfile(rootDir, folderName);

    % Required anatomy for estimating deformation
    t1File = fullfile(folderPath, 'T1.nii');
    if ~exist(t1File, 'file')
        fprintf('Missing T1.nii in %s (skip)\n', folderName);
        continue;
    end

    % Collect resample targets that exist
    resamplePaths = {};
    missing = {};
    for k = 1:numel(resample_files)
        fpath = fullfile(folderPath, resample_files{k});
        if exist(fpath, 'file')
            resamplePaths{end+1,1} = [fpath, ',1']; %#ok<AGROW>
        else
            missing{end+1} = resample_files{k}; %#ok<AGROW>
        end
    end

    if isempty(resamplePaths)
        fprintf('No resample targets found in %s (skip)\n', folderName);
        continue;
    end

    if ~isempty(missing)
        fprintf('Folder %s missing %d file(s): %s\n', ...
            folderName, numel(missing), strjoin(missing, ', '));
        fprintf('  -> Will normalize the files that exist.\n');
    end

    %% ---- BUILD SPM JOB ----
    matlabbatch = [];

    matlabbatch{1}.spm.spatial.normalise.estwrite.subj.vol      = { [t1File, ',1'] };
    matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = resamplePaths;

    % Estimation options (same as your phi script)
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.biasreg  = 0.0001;
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.biasfwhm  = 60;
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.tpm      = {'/apps/spm/spm12/tpm/TPM.nii'};
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.affreg   = 'mni';
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.reg      = [0 0.001 0.5 0.05 0.2];
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.fwhm     = 0;
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.samp     = 3;

    % Writing (resampling) options (same as your phi script)
    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.bb       = [-78 -112 -70; 78 76 85];
    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.vox      = [2 2 2];
    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.interp   = 4;   % B-spline (matches old script)
    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.prefix   = 'w';

    %% ---- RUN SPM JOB ----
    fprintf('Running SPM normalization for folder %s...\n', folderName);
    spm('defaults', 'FMRI');
    spm_jobman('run', matlabbatch);
end

fprintf('Selected part %d of %d completed.\n', partToRun, numParts);
