%% batch_make_theta_phi_to_workspace.m
% Batch-generate theta/phi (radians) from _e.nii direction,
% masked by non-zero voxels in _Jbrain.nii and non-zero |E|.
% Saves outputs into NEW folders inside the current working directory:
%   ./<subject_id>/
%       - T1.nii   (copied from data folder)
%       - T1_tDCSLAB_theta_fromE_maskJbrain.nii
%       - T1_tDCSLAB_phi_fromE_maskJbrain.nii

clear; clc;

%% -------- USER SETTINGS --------
subject_ids = { ...
   '301428'
};

dataBaseDir = '/orange/ruogu.fang/junfu.cheng/SMILE/j_map/new_data/301428/FS6.0_sub-301428_ses-01_T1w/ROAST_11tis_Output_III';

% output root = your current MATLAB working directory
outRoot = pwd;

%% -------- LOOP SUBJECTS --------
for s = 1:numel(subject_ids)
    subject_id = subject_ids{s};
    fprintf('\n=============================\n');
    fprintf('Processing subject: %s\n', subject_id);

    subDataDir = dataBaseDir;

    eFile  = fullfile(subDataDir, 'T1_tDCSLAB_e.nii');
    jFile  = fullfile(subDataDir, 'T1_tDCSLAB_Jbrain.nii');
    t1File = fullfile(subDataDir, 'T1.nii');

    if ~isfile(eFile)
        warning('Missing E file, skipping:\n%s', eFile);
        continue;
    end
    if ~isfile(jFile)
        warning('Missing Jbrain file, skipping:\n%s', jFile);
        continue;
    end
    if ~isfile(t1File)
        warning('Missing T1.nii (will still compute theta/phi):\n%s', t1File);
    end

    % Create output directory in workspace
    outDir = fullfile(outRoot, subject_id);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
        fprintf('Created output folder: %s\n', outDir);
    end

    thetaOut = fullfile(outDir, 'T1_tDCSLAB_theta_fromE_maskJbrain.nii');
    phiOut   = fullfile(outDir, 'T1_tDCSLAB_phi_fromE_maskJbrain.nii');
    t1Out    = fullfile(outDir, 'T1.nii');

    %% ---- COPY T1.nii ----
    if isfile(t1File)
        if ~isfile(t1Out)
            copyfile(t1File, t1Out);
            fprintf('Copied T1.nii to: %s\n', t1Out);
        else
            fprintf('T1.nii already exists in output folder, skipping copy.\n');
        end
    end

    %% ---- LOAD NIfTIs ----
    infoE = niftiinfo(eFile);
    E     = niftiread(infoE);

    infoJ = niftiinfo(jFile);
    Jm    = niftiread(infoJ);

    % Checks
    if numel(infoE.ImageSize) < 4 || infoE.ImageSize(4) ~= 3
        warning('_e.nii not 4D with 3 comps. Got ImageSize: [%s]. Skipping %s', ...
            num2str(infoE.ImageSize), subject_id);
        continue;
    end

    if ~isequal(infoE.ImageSize(1:3), infoJ.ImageSize(1:3))
        warning('Size mismatch E vs Jbrain for %s. E:[%s] J:[%s]. Skipping.', ...
            subject_id, num2str(infoE.ImageSize(1:3)), num2str(infoJ.ImageSize(1:3)));
        continue;
    end

    %% ---- SPLIT COMPONENTS ----
    Ex = double(E(:,:,:,1));
    Ey = double(E(:,:,:,2));
    Ez = double(E(:,:,:,3));
    Jm = double(Jm);

    %% ---- MASK VALID VOXELS ----
    Emag = sqrt(Ex.^2 + Ey.^2 + Ez.^2);

    maskJ = (Jm ~= 0) & ~isnan(Jm);
    maskE = (Emag ~= 0) & ~isnan(Emag);
    maskValid = maskJ & maskE;

    fprintf('Valid voxels (J!=0 & |E|!=0): %d\n', nnz(maskValid));

    %% ---- COMPUTE THETA / PHI (radians) ----
    theta = nan(size(Emag), 'double');
    phi   = nan(size(Emag), 'double');

    c = Ez(maskValid) ./ Emag(maskValid);
    c = max(-1, min(1, c));            % clamp for numeric safety
    theta(maskValid) = acos(c);        % [0, pi]

    phi(maskValid) = atan2(Ey(maskValid), Ex(maskValid)); % [-pi, pi]

    %% ---- WRITE OUTPUTS ----
    % Use Jbrain geometry as template for 3D outputs
    infoOut = infoJ;
    infoOut.ImageSize = infoJ.ImageSize(1:3);
    infoOut.PixelDimensions = infoJ.PixelDimensions(1:3);
    infoOut.Datatype = 'single';
    infoOut.BitsPerPixel = 32;

    infoOut.Description = 'theta_fromE_maskJbrain (rad)';
    niftiwrite(single(theta), thetaOut, infoOut, 'Compressed', false);

    infoOut.Description = 'phi_fromE_maskJbrain (rad)';
    niftiwrite(single(phi), phiOut, infoOut, 'Compressed', false);

    fprintf('Saved:\n  %s\n  %s\n', thetaOut, phiOut);

    %% ---- QUICK RANGE CHECK ----
    if nnz(maskValid) > 0
        fprintf('Theta range (valid): [%g, %g]\n', min(theta(maskValid)), max(theta(maskValid)));
        fprintf('Phi range (valid):   [%g, %g]\n', min(phi(maskValid)), max(phi(maskValid)));
    else
        fprintf('No valid voxels found; theta/phi are all NaN.\n');
    end
end

fprintf('\nAll done.\n');