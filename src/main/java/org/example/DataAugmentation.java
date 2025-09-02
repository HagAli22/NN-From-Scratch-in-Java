package org.example;

import java.util.Random;

/**
 * DataAugmentation class provides various image transformation techniques
 * to increase training data diversity and improve model generalization.
 */
public class DataAugmentation {
    
    private static final Random random = new Random();
    
    /**
     * Apply random rotation to a 28x28 MNIST image
     * @param image Flattened 28x28 image (784 elements)
     * @param maxAngleDegrees Maximum rotation angle in degrees (±)
     * @return Rotated image as flattened array
     */
    public static double[] rotate(double[] image, double maxAngleDegrees) {
        double angle = (random.nextDouble() * 2 - 1) * Math.toRadians(maxAngleDegrees);
        double[][] img2D = reshape1Dto2D(image, 28, 28);
        double[][] rotated = rotateImage(img2D, angle);
        return flatten2Dto1D(rotated);
    }
    
    /**
     * Apply random translation (shift) to image
     * @param image Flattened 28x28 image
     * @param maxShiftPixels Maximum shift in pixels (±)
     * @return Translated image
     */
    public static double[] translate(double[] image, int maxShiftPixels) {
        int shiftX = random.nextInt(2 * maxShiftPixels + 1) - maxShiftPixels;
        int shiftY = random.nextInt(2 * maxShiftPixels + 1) - maxShiftPixels;
        
        double[][] img2D = reshape1Dto2D(image, 28, 28);
        double[][] translated = translateImage(img2D, shiftX, shiftY);
        return flatten2Dto1D(translated);
    }
    
    /**
     * Apply random scaling to image
     * @param image Flattened 28x28 image
     * @param minScale Minimum scale factor
     * @param maxScale Maximum scale factor
     * @return Scaled image
     */
    public static double[] scale(double[] image, double minScale, double maxScale) {
        double scaleFactor = minScale + random.nextDouble() * (maxScale - minScale);
        double[][] img2D = reshape1Dto2D(image, 28, 28);
        double[][] scaled = scaleImage(img2D, scaleFactor);
        return flatten2Dto1D(scaled);
    }
    
    /**
     * Apply elastic distortion (important for handwritten digits)
     * @param image Flattened 28x28 image
     * @param alpha Distortion strength
     * @param sigma Gaussian filter standard deviation
     * @return Elastically distorted image
     */
    public static double[] elasticDistortion(double[] image, double alpha, double sigma) {
        double[][] img2D = reshape1Dto2D(image, 28, 28);
        double[][] distorted = applyElasticDistortion(img2D, alpha, sigma);
        return flatten2Dto1D(distorted);
    }
    
    /**
     * Add Gaussian noise to image
     * @param image Flattened 28x28 image
     * @param noiseStd Standard deviation of noise
     * @return Noisy image
     */
    public static double[] addGaussianNoise(double[] image, double noiseStd) {
        double[] noisy = new double[image.length];
        for (int i = 0; i < image.length; i++) {
            double noise = random.nextGaussian() * noiseStd;
            noisy[i] = Math.max(0.0, Math.min(1.0, image[i] + noise));
        }
        return noisy;
    }
    
    /**
     * Apply random brightness adjustment
     * @param image Flattened 28x28 image
     * @param maxBrightnessChange Maximum brightness change (±)
     * @return Brightness-adjusted image
     */
    public static double[] adjustBrightness(double[] image, double maxBrightnessChange) {
        double brightnessChange = (random.nextDouble() * 2 - 1) * maxBrightnessChange;
        double[] adjusted = new double[image.length];
        
        for (int i = 0; i < image.length; i++) {
            adjusted[i] = Math.max(0.0, Math.min(1.0, image[i] + brightnessChange));
        }
        return adjusted;
    }
    
    /**
     * Apply random contrast adjustment
     * @param image Flattened 28x28 image
     * @param minContrast Minimum contrast factor
     * @param maxContrast Maximum contrast factor
     * @return Contrast-adjusted image
     */
    public static double[] adjustContrast(double[] image, double minContrast, double maxContrast) {
        double contrastFactor = minContrast + random.nextDouble() * (maxContrast - minContrast);
        double[] adjusted = new double[image.length];
        
        for (int i = 0; i < image.length; i++) {
            adjusted[i] = Math.max(0.0, Math.min(1.0, image[i] * contrastFactor));
        }
        return adjusted;
    }
    
    /**
     * Apply multiple random augmentations to an image
     * @param image Original flattened image
     * @param augmentationStrength Strength factor (0.0 = no augmentation, 1.0 = full strength)
     * @return Augmented image
     */
    public static double[] applyRandomAugmentations(double[] image, double augmentationStrength) {
        double[] augmented = image.clone();
        
        // Apply augmentations with probability based on strength
        if (random.nextDouble() < augmentationStrength * 0.7) {
            augmented = rotate(augmented, 15 * augmentationStrength);
        }
        
        if (random.nextDouble() < augmentationStrength * 0.8) {
            augmented = translate(augmented, (int)(3 * augmentationStrength));
        }
        
        if (random.nextDouble() < augmentationStrength * 0.6) {
            double scaleRange = 0.1 * augmentationStrength;
            augmented = scale(augmented, 1.0 - scaleRange, 1.0 + scaleRange);
        }
        
        if (random.nextDouble() < augmentationStrength * 0.5) {
            augmented = elasticDistortion(augmented, 8 * augmentationStrength, 2.0);
        }
        
        if (random.nextDouble() < augmentationStrength * 0.4) {
            augmented = addGaussianNoise(augmented, 0.05 * augmentationStrength);
        }
        
        if (random.nextDouble() < augmentationStrength * 0.3) {
            augmented = adjustBrightness(augmented, 0.1 * augmentationStrength);
        }
        
        if (random.nextDouble() < augmentationStrength * 0.3) {
            augmented = adjustContrast(augmented, 1.0 - 0.2 * augmentationStrength, 
                                     1.0 + 0.2 * augmentationStrength);
        }
        
        return augmented;
    }
    
    /**
     * Generate multiple augmented versions of a single image
     * @param image Original image
     * @param numAugmentations Number of augmented copies to generate
     * @param strength Augmentation strength
     * @return Array of augmented images including original
     */
    public static double[][] generateAugmentedBatch(double[] image, int numAugmentations, double strength) {
        double[][] batch = new double[numAugmentations + 1][];
        batch[0] = image.clone(); // Original image
        
        for (int i = 1; i <= numAugmentations; i++) {
            batch[i] = applyRandomAugmentations(image, strength);
        }
        
        return batch;
    }
    
    // ========== Helper Methods ==========
    
    private static double[][] reshape1Dto2D(double[] flat, int rows, int cols) {
        double[][] reshaped = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                reshaped[i][j] = flat[i * cols + j];
            }
        }
        return reshaped;
    }
    
    private static double[] flatten2Dto1D(double[][] matrix) {
        int rows = matrix.length, cols = matrix[0].length;
        double[] flat = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flat[i * cols + j] = matrix[i][j];
            }
        }
        return flat;
    }
    
    private static double[][] rotateImage(double[][] image, double angle) {
        int size = image.length;
        double[][] rotated = new double[size][size];
        double centerX = size / 2.0, centerY = size / 2.0;
        double cos = Math.cos(angle), sin = Math.sin(angle);
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                // Rotate coordinates
                double dx = x - centerX, dy = y - centerY;
                double srcX = dx * cos + dy * sin + centerX;
                double srcY = -dx * sin + dy * cos + centerY;
                
                rotated[y][x] = bilinearInterpolation(image, srcX, srcY);
            }
        }
        return rotated;
    }
    
    private static double[][] translateImage(double[][] image, int shiftX, int shiftY) {
        int size = image.length;
        double[][] translated = new double[size][size];
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                int srcX = x - shiftX;
                int srcY = y - shiftY;
                
                if (srcX >= 0 && srcX < size && srcY >= 0 && srcY < size) {
                    translated[y][x] = image[srcY][srcX];
                } else {
                    translated[y][x] = 0.0; // Black padding
                }
            }
        }
        return translated;
    }
    
    private static double[][] scaleImage(double[][] image, double scaleFactor) {
        int size = image.length;
        double[][] scaled = new double[size][size];
        double center = size / 2.0;
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double dx = (x - center) / scaleFactor + center;
                double dy = (y - center) / scaleFactor + center;
                
                scaled[y][x] = bilinearInterpolation(image, dx, dy);
            }
        }
        return scaled;
    }
    
    private static double[][] applyElasticDistortion(double[][] image, double alpha, double sigma) {
        int size = image.length;
        double[][] distorted = new double[size][size];
        
        // Generate random displacement fields
        double[][] deltaX = generateGaussianField(size, sigma);
        double[][] deltaY = generateGaussianField(size, sigma);
        
        // Scale by alpha
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                deltaX[i][j] *= alpha;
                deltaY[i][j] *= alpha;
            }
        }
        
        // Apply distortion
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double srcX = x + deltaX[y][x];
                double srcY = y + deltaY[y][x];
                
                distorted[y][x] = bilinearInterpolation(image, srcX, srcY);
            }
        }
        
        return distorted;
    }
    
    private static double[][] generateGaussianField(int size, double sigma) {
        double[][] field = new double[size][size];
        
        // Generate random field
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                field[i][j] = random.nextGaussian();
            }
        }
        
        // Apply Gaussian smoothing (simple approximation)
        return gaussianSmooth(field, sigma);
    }
    
    private static double[][] gaussianSmooth(double[][] field, double sigma) {
        int size = field.length;
        double[][] smoothed = new double[size][size];
        int kernelSize = (int)(3 * sigma) * 2 + 1;
        double[][] kernel = createGaussianKernel(kernelSize, sigma);
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double sum = 0.0;
                double weightSum = 0.0;
                
                for (int ky = 0; ky < kernelSize; ky++) {
                    for (int kx = 0; kx < kernelSize; kx++) {
                        int ny = y + ky - kernelSize / 2;
                        int nx = x + kx - kernelSize / 2;
                        
                        if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
                            double weight = kernel[ky][kx];
                            sum += field[ny][nx] * weight;
                            weightSum += weight;
                        }
                    }
                }
                
                smoothed[y][x] = weightSum > 0 ? sum / weightSum : 0.0;
            }
        }
        
        return smoothed;
    }
    
    private static double[][] createGaussianKernel(int size, double sigma) {
        double[][] kernel = new double[size][size];
        double center = size / 2.0;
        double sum = 0.0;
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double dx = x - center;
                double dy = y - center;
                double value = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                kernel[y][x] = value;
                sum += value;
            }
        }
        
        // Normalize kernel
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                kernel[y][x] /= sum;
            }
        }
        
        return kernel;
    }
    
    private static double bilinearInterpolation(double[][] image, double x, double y) {
        int size = image.length;
        
        // Handle out-of-bounds
        if (x < 0 || x >= size - 1 || y < 0 || y >= size - 1) {
            return 0.0;
        }
        
        int x1 = (int) Math.floor(x);
        int y1 = (int) Math.floor(y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        
        double dx = x - x1;
        double dy = y - y1;
        
        // Bilinear interpolation
        double interpolated = 
            image[y1][x1] * (1 - dx) * (1 - dy) +
            image[y1][x2] * dx * (1 - dy) +
            image[y2][x1] * (1 - dx) * dy +
            image[y2][x2] * dx * dy;
            
        return Math.max(0.0, Math.min(1.0, interpolated));
    }
    
    /**
     * Create augmented training dataset
     * @param originalImages Original training images
     * @param originalLabels Original training labels
     * @param augmentationsPerImage Number of augmented versions per original image
     * @param augmentationStrength Strength of augmentations (0.0 to 1.0)
     * @return AugmentedDataset containing expanded data
     */
    public static AugmentedDataset createAugmentedDataset(double[][] originalImages, 
                                                         double[][] originalLabels, 
                                                         int augmentationsPerImage, 
                                                         double augmentationStrength) {
        
        int originalSize = originalImages.length;
        int newSize = originalSize * (augmentationsPerImage + 1);
        
        double[][] augmentedImages = new double[newSize][];
        double[][] augmentedLabels = new double[newSize][];
        
        int index = 0;
        
        for (int i = 0; i < originalSize; i++) {
            // Add original image
            augmentedImages[index] = originalImages[i].clone();
            augmentedLabels[index] = originalLabels[i].clone();
            index++;
            
            // Add augmented versions
            for (int j = 0; j < augmentationsPerImage; j++) {
                double[] augmentedImage = applyRandomAugmentations(originalImages[i], augmentationStrength);
                augmentedImages[index] = augmentedImage;
                augmentedLabels[index] = originalLabels[i].clone(); // Same label
                index++;
            }
            
            // Progress indicator
            if ((i + 1) % 1000 == 0) {
                System.out.printf("Augmented %d/%d images\n", i + 1, originalSize);
            }
        }
        
        return new AugmentedDataset(augmentedImages, augmentedLabels);
    }
    
    // Helper class to hold augmented dataset
    public static class AugmentedDataset {
        public final double[][] images;
        public final double[][] labels;
        
        public AugmentedDataset(double[][] images, double[][] labels) {
            this.images = images;
            this.labels = labels;
        }
        
        public int getSize() {
            return images.length;
        }
    }
}