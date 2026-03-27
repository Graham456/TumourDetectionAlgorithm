function project()
    
    imagePath = 'projectimg.png'; % 
    maskPath  = 'GTmask.png';

    % Phase 1: Preprocessing
    [img, gt] = preprocess_data(imagePath, maskPath);
    
    % Place to call each function 
    homo_filter_out = homomorphic_filter(img);
    contrast_out = contrast_enhancement(homo_filter_out);
    featureMap = generate_features(contrast_out);
    binaryMask = segment_image(featureMap);
    
    % Display results to verify Phase 1 worked alter for other phases
    figure;
    subplot(1,2,1); imshow(featureMap); title('FeatureMap');
    subplot(1,2,2); imshow(gt);  title('Ground Truth Mask');
    
    
    fprintf('Phase 1 Complete using %s and %s.\n', imagePath, maskPath);
end

% --- LOCAL FUNCTIONS ---

function [imgNorm, gtMask] = preprocess_data(imgP, mskP)
    rawImg = im2double(imread(imgP));
    rawGT  = im2double(imread(mskP));
    if size(rawImg, 3) == 3, rawImg = rgb2gray(rawImg); end % convert to 2D greyscale if images are 2D (RGB)
    if size(rawGT, 3) == 3,  rawGT  = rgb2gray(rawGT);  end
    
    lower = quantile(rawImg(:), 0.01); % remove top and bottom 1% to ignore bright text or black borders
    upper = quantile(rawImg(:), 0.99); 
    
    clipped = max(lower, min(upper, rawImg));
    
    imgNorm = (clipped - lower) / (upper - lower); % force intensity to be between 0 and 1
    
    gtMask = rawGT > 0.5; % binarize ground truth (need to use for IoU calc later)
end

function homo_filter_out = homomorphic_filter(img) % step 2, homomorphic filtering for speckle reduction
    
    imgLog = log(1 + img); % avoid using log(0) 
    
    % Perform 2D FFT and shift the zero-frequency component to the center
    F = fftshift(fft2(imgLog));
    
   
    [M, N] = size(img); % high pass filter kernal
    [U, V] = meshgrid(1:N, 1:M);
    
    % Define the center coordinates
    centerX = floor(N/2) + 1;
    centerY = floor(M/2) + 1;
    
    % D is the distance from the center (frequency origin)
    D = sqrt((U - centerX).^2 + (V - centerY).^2);
    
    % Sigma controls the 'cutoff'. 
    % Lower sigma = smoother/blurrier, Higher sigma = sharper/more noise kept.
    sigma = 30; 
    gammaL = 0.5;
    gammaH = 1.5;

    H = (gammaH - gammaL) * (1 - exp(-(D.^2) / (2 * sigma^2))) + gammaL;
    
    % 4. Apply the Filter
    G = F .* H;
    imgFiltered = real(ifft2(ifftshift(G)));
   
    homo_filter_out = exp(imgFiltered) - 1;
    % Standardize the output to [0, 1] for Phase 3 (Contrast Enhancement)
    if max(homo_filter_out(:)) > min(homo_filter_out(:))
        homo_filter_out = (homo_filter_out - min(homo_filter_out(:))) / (max(homo_filter_out(:)) - min(homo_filter_out(:))); % standardize output to [0,1] for Phase 3 contrast enhancement
    end
end

% Phase 3: Contrast enhancement


function contrast_out = contrast_enhancement(homo_filter_out) 
    % --- Step A: Local Contrast Enhancement ---
    % We use a local mean to adjust intensities
    % Define a local window size (e.g., 15x15)
    h = ones(15,15) / (15*15);
    localMean = imfilter(homo_filter_out, h, 'replicate');
    
    % Enhance: Output = Gain * (Input - localMean) + localMean
    % A gain > 1 increases local contrast
    gain = 1.5;
    localEnhanced = gain * (homo_filter_out - localMean) + localMean;
    % Create a blurred version
    blurKernel = fspecial('gaussian', [15 15], 3);
    blurred = imfilter(localEnhanced, blurKernel, 'replicate');
    
    % Mask = Original - Blurred (this contains only the edges/details)
    mask = localEnhanced - blurred;
    
    % High-boost: Result = Original + k * Mask
    % If k=1, it's standard unsharp masking. If k > 1, it's high-boost.
    k = 2.0; 
    contrast_out = localEnhanced + k * mask;
    
    % Final clip and normalize to keep it in [0, 1]
    contrast_out = max(0, min(1, contrast_out));
end

function featureMap = generate_features(contrast_out)
    % 1. Gradient Magnitude (Sobel Operators)
    % We define the kernels manually as per your project requirements
    Gx = [-1 0 1; -2 0 2; -1 0 1];
    Gy = [1 2 1; 0 0 0; -1 -2 -1];
    
    gradX = imfilter(contrast_out, Gx, 'replicate');
    gradY = imfilter(contrast_out, Gy, 'replicate');
    gradMag = sqrt(gradX.^2 + gradY.^2);
    % Normalize gradient
    gradMag = gradMag / max(gradMag(:));
    
    % Local Variance
    % Measures texture heterogeneity. Window size 7x7 is good for US images.
    h = ones(7,7) / (7*7);
    mu = imfilter(contrast_out, h, 'replicate');
    mu2 = imfilter(contrast_out.^2, h, 'replicate');
    localVar = mu2 - mu.^2; 
    % Normalize variance
    localVar = localVar / max(localVar(:));
    
    % Combined Feature Map
    % We want regions with high variance OR strong edges.
    % We also invert the intensity: Tumors are dark, so (1 - contrast_out) makes them bright
    intensityFeature = 1 - contrast_out;
    
    featureMap = (0.4 * intensityFeature) + (0.3 * gradMag) + (0.3 * localVar);
    featureMap = (featureMap - min(featureMap(:))) / (max(featureMap(:)) - min(featureMap(:)));   
end

function binaryMask = segment_image(featureMap)
    % Calculate the statistical properties of the entire map
    % fMap(:) flattens the 2D matrix into a 1D list for statistics
    avgVal = mean(featureMap(:));
    stdVal = std(featureMap(:));
    
    % Determine the Adaptive Threshold
    % Logic: If a pixel is 'k' standard deviations above the average, 
    % it is likely part of a tumor (the bright regions of the feature map).
    % Start with k = 1.0. Increase it if the mask is too 'noisy'.
    k = 1.0; 
    threshold = avgVal + (k * stdVal);
    
    % 3. Create the Binary Mask (Logical 0 or 1)
    binaryMask = featureMap > threshold;
    
    % Visualizing for the user in the Command Window
    fprintf('Phase 5: Threshold set at %.4f (Mean: %.4f, Std: %.4f)\n', ...
            threshold, avgVal, stdVal);
end



    