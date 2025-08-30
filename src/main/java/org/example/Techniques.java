package org.example;

import java.util.Random;

public class Techniques {

    private static Random random = new Random();

    // Store dropout mask for backpropagation
    private static boolean[][] lastDropoutMask;

    // Store batch norm statistics for backpropagation
    private static double[] lastMean;
    private static double[] lastVariance;
    private static double[][] lastNormalized;

    // ======================= DROPOUT =======================

    public static double[][] dropout(double[][] input, double dropoutRate, boolean isTraining) {
        if (!isTraining || dropoutRate <= 0.0) {
            return input; // No dropout during inference or if rate is 0
        }

        int rows = input.length;
        int cols = input[0].length;
        double[][] result = new double[rows][cols];

        // Create and store dropout mask for backpropagation
        lastDropoutMask = new boolean[rows][cols];

        // Scale factor to maintain expected output during training
        double scale = 1.0 / (1.0 - dropoutRate);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (random.nextDouble() > dropoutRate) {
                    lastDropoutMask[i][j] = true;
                    result[i][j] = input[i][j] * scale;
                } else {
                    lastDropoutMask[i][j] = false;
                    result[i][j] = 0.0;
                }
            }
        }

        return result;
    }

    /**
     * Backpropagation through dropout layer
     */
    public static double[][] dropoutBackward(double[][] dOut, double dropoutRate) {
        if (lastDropoutMask == null) {
            return dOut; // No dropout was applied in forward pass
        }

        int rows = dOut.length;
        int cols = dOut[0].length;
        double[][] dInput = new double[rows][cols];

        double scale = 1.0 / (1.0 - dropoutRate);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (lastDropoutMask[i][j]) {
                    dInput[i][j] = dOut[i][j] * scale;
                } else {
                    dInput[i][j] = 0.0;
                }
            }
        }

        return dInput;
    }

    // ======================= BATCH NORMALIZATION =======================

    public static double[][] batchNormalization(double[][] input, double[][] gamma,
                                                double[][] beta, double epsilon) {
        int batchSize = input.length;
        int features = input[0].length;

        // Calculate mean for each feature
        lastMean = new double[features];
        for (int j = 0; j < features; j++) {
            double sum = 0.0;
            for (int i = 0; i < batchSize; i++) {
                sum += input[i][j];
            }
            lastMean[j] = sum / batchSize;
        }

        // Calculate variance for each feature
        lastVariance = new double[features];
        for (int j = 0; j < features; j++) {
            double sum = 0.0;
            for (int i = 0; i < batchSize; i++) {
                double diff = input[i][j] - lastMean[j];
                sum += diff * diff;
            }
            lastVariance[j] = sum / batchSize;
        }

        // Store normalized values for backprop
        lastNormalized = new double[batchSize][features];

        // Normalize and apply gamma/beta
        double[][] result = new double[batchSize][features];
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < features; j++) {
                // Normalize
                double normalized = (input[i][j] - lastMean[j]) / Math.sqrt(lastVariance[j] + epsilon);
                lastNormalized[i][j] = normalized;

                // Scale and shift
                result[i][j] = gamma[0][j] * normalized + beta[0][j];
            }
        }

        return result;
    }

    /**
     * Batch normalization with default epsilon
     */
    public static double[][] batchNormalization(double[][] input, double[][] gamma, double[][] beta) {
        return batchNormalization(input, gamma, beta, 1e-8);
    }

    /**
     * Backpropagation through batch normalization
     */
    public static double[][] batchNormBackward(double[][] dOut, double[][] originalInput,
                                               double[][] gamma, double[][] beta) {
        if (lastMean == null || lastVariance == null || lastNormalized == null) {
            return dOut; // No batch norm was applied
        }

        int batchSize = dOut.length;
        int features = dOut[0].length;
        double epsilon = 1e-8;

        double[][] dInput = new double[batchSize][features];

        for (int j = 0; j < features; j++) {
            double stdDev = Math.sqrt(lastVariance[j] + epsilon);

            // Calculate intermediate gradients
            double dVar = 0.0;
            double dMean = 0.0;

            // dVar calculation
            for (int i = 0; i < batchSize; i++) {
                dVar += dOut[i][j] * gamma[0][j] * (originalInput[i][j] - lastMean[j]) * (-0.5) * Math.pow(stdDev, -3);
            }

            // dMean calculation
            for (int i = 0; i < batchSize; i++) {
                dMean += dOut[i][j] * gamma[0][j] * (-1.0 / stdDev);
            }
            dMean += dVar * (-2.0 / batchSize) * sumColumn(originalInput, lastMean[j], j);

            // Final dInput calculation
            for (int i = 0; i < batchSize; i++) {
                dInput[i][j] = dOut[i][j] * gamma[0][j] / stdDev +
                        dVar * 2.0 * (originalInput[i][j] - lastMean[j]) / batchSize +
                        dMean / batchSize;
            }
        }

        return dInput;
    }

    /**
     * Compute gamma gradients for batch normalization
     */
    public static double[][] computeGammaGradient(double[][] dOut, double[][] batchNormOutput) {
        if (lastNormalized == null) {
            return new double[1][dOut[0].length]; // Return zeros if no batch norm
        }

        int features = dOut[0].length;
        int batchSize = dOut.length;
        double[][] dGamma = new double[1][features];

        for (int j = 0; j < features; j++) {
            double sum = 0.0;
            for (int i = 0; i < batchSize; i++) {
                sum += dOut[i][j] * lastNormalized[i][j];
            }
            dGamma[0][j] = sum;
        }

        return dGamma;
    }

    /**
     * Compute beta gradients for batch normalization
     */
    public static double[][] computeBetaGradient(double[][] dOut) {
        int features = dOut[0].length;
        int batchSize = dOut.length;
        double[][] dBeta = new double[1][features];

        for (int j = 0; j < features; j++) {
            double sum = 0.0;
            for (int i = 0; i < batchSize; i++) {
                sum += dOut[i][j];
            }
            dBeta[0][j] = sum;
        }

        return dBeta;
    }

    /**
     * Helper method for batch norm backprop
     */
    private static double sumColumn(double[][] matrix, double mean, int col) {
        double sum = 0.0;
        for (int i = 0; i < matrix.length; i++) {
            sum += (matrix[i][col] - mean);
        }
        return sum;
    }

    // ======================= REGULARIZATION =======================

    public static double l2Regularization(double[][] weights, double lambda) {
        double sum = 0.0;

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                sum += weights[i][j] * weights[i][j];
            }
        }

        return lambda * sum * 0.5; // 0.5 factor is standard in L2 reg
    }

    /**
     * Calculate L1 regularization loss
     * @param weights Weight matrix
     * @param lambda Regularization strength
     * @return L1 penalty value
     */
    public static double l1Regularization(double[][] weights, double lambda) {
        double sum = 0.0;

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                sum += Math.abs(weights[i][j]);
            }
        }

        return lambda * sum;
    }

    /**
     * Calculate total regularization loss for multiple weight matrices
     * @param weights Array of weight matrices
     * @param lambda Regularization strength
     * @param useL1 True for L1, false for L2
     * @return Total regularization penalty
     */
    public static double totalRegularization(double[][][] weights, double lambda, boolean useL1) {
        double totalLoss = 0.0;

        for (double[][] weightMatrix : weights) {
            if (useL1) {
                totalLoss += l1Regularization(weightMatrix, lambda);
            } else {
                totalLoss += l2Regularization(weightMatrix, lambda);
            }
        }

        return totalLoss;
    }

    /**
     * Apply weight decay (L2 regularization gradient)
     * @param weights Weight matrix to apply decay to
     * @param lambda Regularization strength
     * @return Gradient matrix for weight decay
     */
    public static double[][] weightDecayGradient(double[][] weights, double lambda) {
        int rows = weights.length;
        int cols = weights[0].length;
        double[][] gradient = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                gradient[i][j] = lambda * weights[i][j];
            }
        }

        return gradient;
    }

    // ======================= UTILITY METHODS =======================



    /**
     * Set random seed for reproducible dropout
     */
    public static void setRandomSeed(long seed) {
        random = new Random(seed);
    }

    /**
     * Initialize gamma parameters for batch normalization (usually to 1.0)
     */
    public static double[][] initializeGamma(int features) {
        double[][] gamma = new double[1][features];
        for (int i = 0; i < features; i++) {
            gamma[0][i] = 1.0;
        }
        return gamma;
    }

    /**
     * Initialize beta parameters for batch normalization (usually to 0.0)
     */
    public static double[][] initializeBeta(int features) {
        double[][] beta = new double[1][features];
        for (int i = 0; i < features; i++) {
            beta[0][i] = 0.0;
        }
        return beta;
    }

    /**
     * Clear stored values (call this between batches if needed)
     */
    public static void clearCache() {
        lastDropoutMask = null;
        lastMean = null;
        lastVariance = null;
        lastNormalized = null;
    }
}