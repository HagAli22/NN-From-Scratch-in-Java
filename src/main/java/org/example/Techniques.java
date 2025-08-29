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




}