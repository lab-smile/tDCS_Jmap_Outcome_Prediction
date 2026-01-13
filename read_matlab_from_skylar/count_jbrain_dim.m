%% Robust NIfTI dimension & voxel count script

% ---- SET FULL PATH TO FILE ----
niiFile = '/orange/ruogu.fang/trials/ACT/roast_11_data/100283/T1_tDCSLAB_Jbrain.nii';

% ---- CHECK FILE EXISTS ----
if ~isfile(niiFile)
    error('NIfTI file not found:\n%s', niiFile);
end

% ---- READ NIfTI ----
info = niftiinfo(niiFile);
img  = niftiread(info);

% ---- DIMENSIONS ----
sz = size(img);

fprintf('\nFile: %s\n', niiFile);
fprintf('Image dimensions (size): [%s]\n', num2str(sz));
fprintf('Header ImageSize:        [%s]\n', num2str(info.ImageSize));

% ---- VOXEL COUNTS ----
nTotal = numel(img);

imgD = double(img);  % safe casting

nNaN     = nnz(isnan(imgD));
nZero    = nnz(imgD == 0 & ~isnan(imgD));
nNonZero = nnz(imgD ~= 0 & ~isnan(imgD));

fprintf('\nVoxel statistics:\n');
fprintf('Total voxels:    %d\n', nTotal);
fprintf('NaN voxels:      %d\n', nNaN);
fprintf('Zero voxels:     %d\n', nZero);
fprintf('Non-zero voxels: %d\n', nNonZero);

fprintf('Check sum (Zero + Non-zero + NaN): %d\n', ...
        nZero + nNonZero + nNaN);

% ---- OPTIONAL INFO ----
fprintf('\nDatatype: %s\n', class(img));
fprintf('Value range (ignoring NaN): [%g, %g]\n', ...
        min(imgD(~isnan(imgD))), max(imgD(~isnan(imgD))));
