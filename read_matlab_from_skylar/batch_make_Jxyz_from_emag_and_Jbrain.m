%% batch_make_Jxyz_from_emag_and_Jbrain.m
% For each subject:
%   - copies T1.nii (if exists) into ./<subject_id>/
%   - copies T1_tDCSLAB_Jbrain.nii into ./<subject_id>/
%   - reads T1_tDCSLAB_emag.nii (direction field, 4D, 3 comps)
%   - creates Jbrain component maps: Jx Jy Jz using direction * J intensity
% Outputs into NEW folders inside current working directory:
%   ./<subject_id>/
%       - T1.nii (copied if exists)
%       - T1_tDCSLAB_Jbrain.nii (copied)
%       - T1_tDCSLAB_Jbrain_x_fromEmag.nii
%       - T1_tDCSLAB_Jbrain_y_fromEmag.nii
%       - T1_tDCSLAB_Jbrain_z_fromEmag.nii

clear; clc;

%% -------- USER SETTINGS --------
subject_ids = { ...
    '105256', ...
    '105601', ...
    '105971', ...
    '107802', ...
    '109021', ...
    '202808', ...
    '203846', ...
    '110081', ...
    '115991', ...
    '116036', ...
    '202251', ...
    '300700', ... % '301428'
    '303009', ...
    '303346', ...
    '301112', ...
    '301513', ...
    '301538', ...
    '303293', ...
    '303673', ...
};

dataBaseDir = '/orange/ruogu.fang/trials/ACT/roast_11_data';

% output root = your current MATLAB working directory
outRoot = pwd;

%% -------- LOOP SUBJECTS --------
for s = 1:numel(subject_ids)
    subject_id = subject_ids{s};
    fprintf('\n=============================\n');
    fprintf('Processing subject: %s\n', subject_id);

    subDataDir = fullfile(dataBaseDir, subject_id);

    emagFile = fullfile(subDataDir, 'T1_tDCSLAB_e.nii');      % direction field (4D, 3 comps)
    jFile    = fullfile(subDataDir, 'T1_tDCSLAB_Jbrain.nii');    % scalar intensity
    t1File   = fullfile(subDataDir, 'T1.nii');

    if ~isfile(emagFile)
        warning('Missing emag (direction) file, skipping:\n%s', emagFile);
        continue;
    end
    if ~isfile(jFile)
        warning('Missing Jbrain file, skipping:\n%s', jFile);
        continue;
    end
    if ~isfile(t1File)
        warning('Missing T1.nii (will still compute Jx/Jy/Jz):\n%s', t1File);
    end

    %% ---- CREATE OUTPUT DIRECTORY ----
    outDir = fullfile(outRoot, subject_id);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
        fprintf('Created output folder: %s\n', outDir);
    end

    t1Out = fullfile(outDir, 'T1.nii');
    jOut  = fullfile(outDir, 'T1_tDCSLAB_Jbrain.nii');

    jxOut = fullfile(outDir, 'T1_tDCSLAB_Jbrain_x_fromEmag.nii');
    jyOut = fullfile(outDir, 'T1_tDCSLAB_Jbrain_y_fromEmag.nii');
    jzOut = fullfile(outDir, 'T1_tDCSLAB_Jbrain_z_fromEmag.nii');

    %% ---- COPY T1.nii ----
    if isfile(t1File)
        if ~isfile(t1Out)
            copyfile(t1File, t1Out);
            fprintf('Copied T1.nii to: %s\n', t1Out);
        else
            fprintf('T1.nii already exists in output folder, skipping copy.\n');
        end
    end

    %% ---- COPY Jbrain ----
    if ~isfile(jOut)
        copyfile(jFile, jOut);
        fprintf('Copied Jbrain to: %s\n', jOut);
    else
        fprintf('Jbrain already exists in output folder, skipping copy.\n');
    end

    %% ---- LOAD NIfTIs ----
    infoE = niftiinfo(emagFile);
    E     = niftiread(infoE);

    infoJ = niftiinfo(jFile);
    J     = niftiread(infoJ);

    % Checks: direction field must be 4D with 3 comps
    if numel(infoE.ImageSize) < 4 || infoE.ImageSize(4) ~= 3
        warning('emag.nii not 4D with 3 comps. Got ImageSize: [%s]. Skipping %s', ...
            num2str(infoE.ImageSize), subject_id);
        continue;
    end

    if ~isequal(infoE.ImageSize(1:3), infoJ.ImageSize(1:3))
        warning('Size mismatch emag vs Jbrain for %s. E:[%s] J:[%s]. Skipping.', ...
            subject_id, num2str(infoE.ImageSize(1:3)), num2str(infoJ.ImageSize(1:3)));
        continue;
    end

    %% ---- SPLIT COMPONENTS ----
    Ex = double(E(:,:,:,1));
    Ey = double(E(:,:,:,2));
    Ez = double(E(:,:,:,3));
    J  = double(J);

    %% ---- MASK VALID VOXELS ----
    Emag = sqrt(Ex.^2 + Ey.^2 + Ez.^2);

    maskJ = (J ~= 0) & ~isnan(J);
    maskE = (Emag ~= 0) & ~isnan(Emag);
    maskValid = maskJ & maskE;

    fprintf('Valid voxels (J!=0 & |E|!=0): %d\n', nnz(maskValid));

    %% ---- COMPUTE UNIT DIRECTION + J COMPONENTS ----
    ux = nan(size(J), 'double');
    uy = nan(size(J), 'double');
    uz = nan(size(J), 'double');

    ux(maskValid) = Ex(maskValid) ./ Emag(maskValid);
    uy(maskValid) = Ey(maskValid) ./ Emag(maskValid);
    uz(maskValid) = Ez(maskValid) ./ Emag(maskValid);

    Jx = nan(size(J), 'double');
    Jy = nan(size(J), 'double');
    Jz = nan(size(J), 'double');

    Jx(maskValid) = J(maskValid) .* ux(maskValid);
    Jy(maskValid) = J(maskValid) .* uy(maskValid);
    Jz(maskValid) = J(maskValid) .* uz(maskValid);

    %% ---- WRITE OUTPUTS ----
    % Use Jbrain geometry as template for 3D outputs
    infoOut = infoJ;
    infoOut.ImageSize = infoJ.ImageSize(1:3);
    infoOut.PixelDimensions = infoJ.PixelDimensions(1:3);
    infoOut.Datatype = 'single';
    infoOut.BitsPerPixel = 32;

    infoOut.Description = 'Jbrain_x = Jbrain * (Ex/|E|) from emag (masked J!=0 & |E|!=0)';
    niftiwrite(single(Jx), jxOut, infoOut, 'Compressed', false);

    infoOut.Description = 'Jbrain_y = Jbrain * (Ey/|E|) from emag (masked J!=0 & |E|!=0)';
    niftiwrite(single(Jy), jyOut, infoOut, 'Compressed', false);

    infoOut.Description = 'Jbrain_z = Jbrain * (Ez/|E|) from emag (masked J!=0 & |E|!=0)';
    niftiwrite(single(Jz), jzOut, infoOut, 'Compressed', false);

    fprintf('Saved:\n  %s\n  %s\n  %s\n', jxOut, jyOut, jzOut);

    %% ---- QUICK RANGE CHECK ----
    if nnz(maskValid) > 0
        fprintf('J range (valid):  [%g, %g]\n', min(J(maskValid)), max(J(maskValid)));
        fprintf('Jx range (valid): [%g, %g]\n', min(Jx(maskValid)), max(Jx(maskValid)));
        fprintf('Jy range (valid): [%g, %g]\n', min(Jy(maskValid)), max(Jy(maskValid)));
        fprintf('Jz range (valid): [%g, %g]\n', min(Jz(maskValid)), max(Jz(maskValid)));
    else
        fprintf('No valid voxels found; Jx/Jy/Jz are all NaN.\n');
    end
end

fprintf('\nAll done.\n');
