% registration_all_auto_part_theta.m

rootDir = '/orange/ruogu.fang/junfu.cheng/SMILE/j_map/j_map_direction/read_matlab_from_skylar';
folders = dir(rootDir);
phi_file_name = 'T1_tDCSLAB_theta_fromE_maskJbrain.nii';

% Parameters to control which chunk to process
numParts = 2;      % Total number of parts to split the job
partToRun = 2;      % Change this to select which part to run (1-based index)

% Get only the numeric folders
folderList = folders([folders.isdir] & ~ismember({folders.name}, {'.', '..'}) & ~cellfun(@isempty, regexp({folders.name}, '^\d+$', 'once')));
% Sort folder names numerically
folderNames = {folderList.name};
[~, sortIdx] = sort(str2double(folderNames));
folderList = folderList(sortIdx);
% Compute partition indices
totalFolders = length(folderList);
foldersPerPart = ceil(totalFolders / numParts);
startIdx = (partToRun - 1) * foldersPerPart + 1;
endIdx = min(partToRun * foldersPerPart, totalFolders);

% Loop through each folder
for i = startIdx:endIdx
    folderName = folderList(i).name;
    folderPath = fullfile(rootDir, folderName);

    % Check required files
    t1File = fullfile(folderPath, 'T1.nii');
    jbrainFile = fullfile(folderPath, phi_file_name);

    if ~(exist(t1File, 'file') && exist(jbrainFile, 'file'))
        fprintf('Missing required files in %s\n', folderName);
        continue;
    end

    % Build the matlabbatch job
    matlabbatch = [];

    matlabbatch{1}.spm.spatial.normalise.estwrite.subj.vol = { [t1File, ',1'] };
    matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = { [jbrainFile, ',1'] };
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.biasreg = 0.0001;
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.biasfwhm = 60;
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.tpm = {'/apps/spm/spm12/tpm/TPM.nii'};
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.affreg = 'mni';
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.fwhm = 0;
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.samp = 3;
    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.bb = [-78 -112 -70; 78 76 85];
    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.vox = [2 2 2];
    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.interp = 4;
    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.prefix = 'w';

    % Run SPM job
    fprintf('Running SPM normalization for folder %s...\n', folderName);
    spm('defaults', 'FMRI');
    spm_jobman('run', matlabbatch);
end

fprintf('Selected part %d of %d completed.\n', partToRun, numParts);
