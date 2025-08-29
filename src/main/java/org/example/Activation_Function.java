package org.example;

public class Activation_Function {
    // ---------------- ReLU ---------------- //
    public static double[][] relu(double[][] input) {
        int rows = input.length, cols = input[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i][j] = Math.max(0, input[i][j]);

        return result;
    }
    // ---------------- Batch Normalization After Relu ---------------- //
    public static double[][] batchNormAfterRelu(double[][] input, double epsilon) {
        // Step 1: Apply ReLU first
        double[][] reluOutput = relu(input);

        int rows = reluOutput.length, cols = reluOutput[0].length;
        double[][] result = new double[rows][cols];

        // Step 2: Apply Batch Normalization
        for (int j = 0; j < cols; j++) {
            // Extract column
            double[] column = new double[rows];
            for (int i = 0; i < rows; i++) {
                column[i] = reluOutput[i][j];
            }

            // Compute mean
            double mean = 0;
            for (double v : column) mean += v;
            mean /= rows;

            // Compute variance
            double variance = 0;
            for (double v : column) variance += Math.pow(v - mean, 2);
            variance /= rows;

            double stdDev = Math.sqrt(variance + epsilon);

            // Normalize
            for (int i = 0; i < rows; i++) {
                result[i][j] = (reluOutput[i][j] - mean) / stdDev;
            }
        }

        return result;
    }

    // ---------------- Softmax ---------------- //
    public static double[][] softmax(double[][] input) {
        int rows = input.length, cols = input[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            // 1. Find max for numerical stability
            double maxLogit = Matrix_Operations.findMax(input[i]);

            // 2. Compute exp(x - max) and sum
            double expSum = Matrix_Operations.sumExp(input[i], maxLogit, result[i]);

            // 3. Normalize
            for (int j = 0; j < cols; j++) {
                result[i][j] /= expSum;
            }
        }

        return result;
    }





}
