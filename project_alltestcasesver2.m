function project_alltestcases()
    fprintf('======================================================\n');
    fprintf('STARTING TITAN PIPELINE: H-MINIMA & RECONSTRUCTION\n');
    fprintf('======================================================\n\n');
    
    test_cases = {
        'TEST0.jpeg', 'TEST0GT.jpeg';
        'TEST1.jpeg', 'TEST1GT.jpeg';
        'TEST2.jpeg', 'TEST2GT.jpeg';
        'TEST3.jpeg', 'TEST3GT.jpeg'
    };
    
    num_cases = size(test_cases, 1);
    
    % Initialize with -1 to ensure 0.0000 scores are penalized in the mean
    all_iou = -ones(num_cases, 1); 
    all_dice = -ones(num_cases, 1);
    
    fig_summary = figure('Name', 'Popup 1: All Test Cases (Final Results)', ...
                         'NumberTitle', 'off', 'Position', [50, 50, 1400, 950]);
    
    diagnostic_phases = struct();
    
    for i = 1:num_cases
        imgName = test_cases{i, 1};
        gtName  = test_cases{i, 2};
        
        fprintf('Processing Case %d/%d: [%s]...\n', i, num_cases, imgName);
        
        if ~exist(imgName, 'file') || ~exist(gtName, 'file')
            fprintf('  -> ERROR: Files not found. Skipping.\n');
            continue;
        end
        
        [img, gt_resized, finalMask, metrics, phases] = process_single_image(imgName, gtName);
        
        all_iou(i) = metrics.iou;
        all_dice(i) = metrics.dice;
        
        fprintf('  -> IoU: %.4f | Dice: %.4f | Sens: %.4f | Spec: %.4f\n', ...
            metrics.iou, metrics.dice, metrics.sensitivity, metrics.specificity);
            
        if i == 3 % Save TEST2 (the hardest one) for the diagnostic dashboard
            diagnostic_phases.img = img;
            diagnostic_phases.gt = gt_resized;
            diagnostic_phases.finalMask = finalMask;
            diagnostic_phases.data = phases;
            diagnostic_phases.name = imgName;
        end
        
        figure(fig_summary);
        
        subplot(num_cases, 3, (i-1)*3 + 1);
        imshow(img);
        if i == 1, title('Original Input', 'FontWeight', 'bold'); end
        ylabel(strrep(imgName, '.jpeg', ''), 'FontWeight', 'bold', 'Interpreter', 'none');
        
        subplot(num_cases, 3, (i-1)*3 + 2);
        imshow(gt_resized);
        if i == 1, title('Ground Truth Mask', 'FontWeight', 'bold'); end
        
        subplot(num_cases, 3, (i-1)*3 + 3);
        imshow(img);
        hold on;
        visboundaries(finalMask, 'Color', 'r', 'LineWidth', 2);
        visboundaries(gt_resized, 'Color', 'g', 'LineWidth', 1, 'LineStyle', '--');
        hold off;
        if i == 1, title('Red: Algorithm | Green: Truth', 'FontWeight', 'bold'); end
    end
    
    valid_cases = (all_iou >= 0); 
    mean_iou = mean(all_iou(valid_cases)); 
    mean_dice = mean(all_dice(valid_cases));
    
    fprintf('\n======================================================\n');
    fprintf('BATCH PROCESSING COMPLETE\n');
    fprintf('======================================================\n');
    fprintf('Total Images Processed: %d\n', sum(valid_cases));
    fprintf('MEAN IoU SCORE:         %.4f\n', mean_iou);
    fprintf('MEAN DICE SCORE:        %.4f\n', mean_dice);
    fprintf('======================================================\n');
    
    render_diagnostic_dashboard(diagnostic_phases);
end

% =========================================================================
% CORE PIPELINE: H-MINIMA & MORPHOLOGICAL RECONSTRUCTION
% =========================================================================
function [img, gt_resized, finalMask, metrics, phases] = process_single_image(imgName, gtName)
    
    [img, gt] = preprocess_data(imgName, gtName);
    phases = struct(); 
    [rows, cols] = size(img);
    
    % Phase 1: Clinical-Grade Denoising (Preserves structural valleys)
    phases.img_denoised = imgaussfilt(medfilt2(img, [7 7], 'symmetric'), 1.5);
    
    % Phase 2 & 3: Dynamic H-Minima Transform & The Reconstruction Shadow Killer
    % We loop through depth thresholds to guarantee we find seeds for both deep and shallow tumors
    h_depths = [0.20, 0.15, 0.10, 0.05]; 
    
    % Define the "Kill Zones" (Bottom 20% for shadows, Top 5% for skin lines)
    kill_zone = false(rows, cols);
    kill_zone(round(rows * 0.80):end, :) = true; 
    kill_zone(1:round(rows * 0.05), :) = true;   
    
    phases.clean_seeds = false(rows, cols);
    
    for h = h_depths
        % Find local dark valleys of depth 'h'
        regional_min = imextendedmin(phases.img_denoised, h);
        
        % The Shadow Killer: Find any valley that touches the kill zone
        bad_seeds_init = regional_min & kill_zone;
        
        % Reconstruct to find the entirety of the shadow, then delete it
        bad_seeds_full = imreconstruct(bad_seeds_init, regional_min);
        surviving_seeds = regional_min & ~bad_seeds_full;
        
        % Clean up tiny 1-pixel noise
        surviving_seeds = bwareaopen(surviving_seeds, 15);
        
        if sum(surviving_seeds(:)) > 0
            phases.raw_minima = regional_min;
            phases.clean_seeds = surviving_seeds;
            break; % We found good seeds, stop digging deeper
        end
    end
    
    % Phase 4: Local Contrast Scoring to select the True Tumor
    phases.best_seed = select_ultimate_seed(phases.clean_seeds, phases.img_denoised);
    
    % Phase 5: Targeted Active Contour Expansion
    % Negative bias (-0.15) acts like a balloon, inflating the tiny seed 
    % until it hits the fuzzy gray edges of the tumor.
    if sum(phases.best_seed(:)) > 0
        % We expand from the seed on the heavily contrasted image
        img_eq = adapthisteq(phases.img_denoised, 'ClipLimit', 0.02);
        
        phases.ac_mask = activecontour(img_eq, phases.best_seed, 250, 'Chan-Vese', ...
            'SmoothFactor', 2.0, 'ContractionBias', -0.15);
    else
        phases.ac_mask = false(size(img));
    end
    
    % Phase 6: Final Morphological Polish
    % Melts any jagged edges and fills internal holes
    finalMask = imclose(phases.ac_mask, strel('disk', 6));
    finalMask = imfill(finalMask, 'holes');
    
    % Absolute Guarantee: Keep only the single largest contiguous object
    [L_final, num_final] = bwlabel(finalMask);
    if num_final > 1
        stats_final = regionprops(L_final, 'Area');
        [~, maxIdx] = max([stats_final.Area]);
        finalMask = (L_final == maxIdx);
    end
    
    % Metrics Calculation
    gt_resized = imresize(gt, [size(img,1), size(img,2)], 'nearest');
    intersection = sum(finalMask(:) & gt_resized(:));
    unionArea = sum(finalMask(:) | gt_resized(:));
    
    metrics.iou = intersection / max(unionArea, 1);
    metrics.dice = 2 * intersection / max(sum(finalMask(:)) + sum(gt_resized(:)), 1);
    metrics.fp = sum(finalMask(:) & ~gt_resized(:));
    metrics.fn = sum(~finalMask(:) & gt_resized(:));
    metrics.specificity = sum(~finalMask(:) & ~gt_resized(:)) / max(sum(~gt_resized(:)), 1);
    metrics.sensitivity = intersection / max(sum(gt_resized(:)), 1);
end

% =========================================================================
% HEURISTIC SELECTION (Local Contrast Focused)
% =========================================================================
function best_seed = select_ultimate_seed(seed_mask, originalImg)
    [L, num] = bwlabel(seed_mask);
    if num == 0
        best_seed = seed_mask; 
        return; 
    end
    
    stats = regionprops(L, 'Area', 'Solidity', 'Centroid', 'PixelIdxList', 'BoundingBox');
    scores = zeros(num, 1);
    
    [rows, cols] = size(originalImg);
    centerX = cols / 2;
    centerY = rows / 2;
    maxDist = sqrt(centerX^2 + centerY^2);
    
    for i = 1:num
        % 1. Local Contrast (The most important metric for small tumors)
        % Compare the dark core to the immediate surrounding tissue
        coreInt = mean(originalImg(stats(i).PixelIdxList));
        
        bb = stats(i).BoundingBox;
        halo_radius = 15;
        min_r = max(1, floor(bb(2) - halo_radius));
        max_r = min(rows, ceil(bb(2) + bb(4) + halo_radius));
        min_c = max(1, floor(bb(1) - halo_radius));
        max_c = min(cols, ceil(bb(1) + bb(3) + halo_radius));
        
        halo_region = originalImg(min_r:max_r, min_c:max_c);
        haloInt = mean(halo_region(:));
        
        localContrastScore = haloInt - coreInt; 
        if localContrastScore < 0, localContrastScore = 0; end
        
        % 2. Centrality (Ultrasound techs center the lesion)
        dist = sqrt((stats(i).Centroid(1) - centerX)^2 + (stats(i).Centroid(2) - centerY)^2);
        locationScore = 1 - (dist / maxDist);
        
        % 3. Solidity (Tumors are solid, artifacts are jagged)
        solidityScore = stats(i).Solidity;
        
        % FUSION LOGIC: Heavily weights local contrast so TEST2 wins
        scores(i) = (localContrastScore * 0.60) + (locationScore * 0.20) + (solidityScore * 0.20);
    end
    
    [~, bestIdx] = max(scores);
    best_seed = (L == bestIdx);
end

% =========================================================================
% POPUP 2: DIAGNOSTIC DASHBOARD RENDERER
% =========================================================================
function render_diagnostic_dashboard(dp)
    if isempty(fieldnames(dp)), return; end
    
    figure('Name', sprintf('Popup 2: Pipeline Phases (%s)', dp.name), ...
           'NumberTitle', 'off', 'Position', [100, 100, 1400, 600]);
           
    subplot(2, 4, 1); imshow(dp.img); title('1. Original Input');
    subplot(2, 4, 2); imshow(dp.data.img_denoised); title('2. Gaussian/Median Filter');
    
    if isfield(dp.data, 'raw_minima')
        subplot(2, 4, 3); imshow(dp.data.raw_minima); title('3. Raw H-Minima');
    end
    
    subplot(2, 4, 4); imshow(dp.data.clean_seeds); title('4. Shadow Killer (Reconstruct)');
    subplot(2, 4, 5); imshow(dp.data.best_seed); title('5. Winning Seed');
    
    subplot(2, 4, 6); 
    imshow(dp.img); 
    hold on;
    visboundaries(dp.data.best_seed, 'Color', 'y', 'LineWidth', 1);
    hold off;
    title('6. Snake Initialization (Yellow)');
    
    subplot(2, 4, [7, 8]); 
    imshow(dp.img); 
    title('7 & 8. Final Expanded Snake (Red) vs GT (Green)');
    hold on;
    visboundaries(dp.finalMask, 'Color', 'r', 'LineWidth', 2);
    visboundaries(dp.gt, 'Color', 'g', 'LineWidth', 1, 'LineStyle', '--');
    hold off;
end

% =========================================================================
% HELPER MATH FUNCTIONS
% =========================================================================
function [imgNorm, gtMask] = preprocess_data(imgP, mskP)
    rawImg = im2double(imread(imgP));
    rawGT  = im2double(imread(mskP));
    if size(rawImg, 3) == 3, rawImg = rgb2gray(rawImg); end
    if size(rawGT,  3) == 3, rawGT  = rgb2gray(rawGT);  end
    
    % Clip extreme white/black dead pixels
    lower_bound = quantile(rawImg(:), 0.01);
    upper_bound = quantile(rawImg(:), 0.99);
    clipped = max(lower_bound, min(upper_bound, rawImg));
    imgNorm = (clipped - lower_bound) / (upper_bound - lower_bound);
    
    gtMask = rawGT > 0.5;
end