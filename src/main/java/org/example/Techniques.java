package org.example;

import java.util.Random;

public class Techniques {

    private static Random random = new Random();



    // ======================= DROPOUT =======================


    public static double[][] dropout(double[][] input, double dropoutRate, boolean isTraining) {
        if (!isTraining || dropoutRate <= 0.0) {
            return input; // No dropout during inference or if rate is 0
        }

        int rows = input.length;
        int cols = input[0].length;
        double[][] result = new double[rows][cols];

        // Scale factor to maintain expected output during training
        double scale = 1.0 / (1.0 - dropoutRate);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (random.nextDouble() > dropoutRate) {
                    result[i][j] = input[i][j] * scale;
                } else {
                    result[i][j] = 0.0;
                }
            }
        }

        return result;
    }



    // ======================= BATCH NORMALIZATION =======================


    public static double[][] batchNormalization(double[][] input, double[][] gamma,
                                                double[][] beta, double epsilon) {
        int batchSize = input.length;
        int features = input[0].length;

        // Calculate mean for each feature
        double[] mean = new double[features];
        for (int j = 0; j < features; j++) {
            double sum = 0.0;
            for (int i = 0; i < batchSize; i++) {
                sum += input[i][j];
            }
            mean[j] = sum / batchSize;
        }

        // Calculate variance for each feature
        double[] variance = new double[features];
        for (int j = 0; j < features; j++) {
            double sum = 0.0;
            for (int i = 0; i < batchSize; i++) {
                double diff = input[i][j] - mean[j];
                sum += diff * diff;
            }
            variance[j] = sum / batchSize;
        }

        // Normalize and apply gamma/beta
        double[][] result = new double[batchSize][features];
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < features; j++) {
                // Normalize
                double normalized = (input[i][j] - mean[j]) / Math.sqrt(variance[j] + epsilon);
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
}