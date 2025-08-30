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

    // ---------------- ReLU Derivative From Output ---------------- //
    public static double[][] reluDerivativeFromOutput(double[][] reluOutput) {
        int rows = reluOutput.length, cols = reluOutput[0].length;
        double[][] derivative = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // If ReLU output > 0 → derivative = 1, otherwise = 0
                derivative[i][j] = reluOutput[i][j] > 0 ? 1.0 : 0.0;
            }
        }

        return derivative;
    }

    // ---------------- ReLU Derivative From Net Input (ADDED) ---------------- //
    public static double[][] reluDerivativeFromNet(double[][] netInput) {
        int rows = netInput.length, cols = netInput[0].length;
        double[][] derivative = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // If net input > 0 → derivative = 1, otherwise = 0
                derivative[i][j] = netInput[i][j] > 0 ? 1.0 : 0.0;
            }
        }

        return derivative;
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

    // ---------------- Sigmoid (Optional Addition) ---------------- //
    public static double[][] sigmoid(double[][] input) {
        int rows = input.length, cols = input[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = 1.0 / (1.0 + Math.exp(-input[i][j]));
            }
        }

        return result;
    }

    // ---------------- Sigmoid Derivative (Optional Addition) ---------------- //
    public static double[][] sigmoidDerivative(double[][] sigmoidOutput) {
        int rows = sigmoidOutput.length, cols = sigmoidOutput[0].length;
        double[][] derivative = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // derivative = sigmoid(x) * (1 - sigmoid(x))
                derivative[i][j] = sigmoidOutput[i][j] * (1.0 - sigmoidOutput[i][j]);
            }
        }

        return derivative;
    }

    // ---------------- Tanh (Optional Addition) ---------------- //
    public static double[][] tanh(double[][] input) {
        int rows = input.length, cols = input[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.tanh(input[i][j]);
            }
        }

        return result;
    }

    // ---------------- Tanh Derivative (Optional Addition) ---------------- //
    public static double[][] tanhDerivative(double[][] tanhOutput) {
        int rows = tanhOutput.length, cols = tanhOutput[0].length;
        double[][] derivative = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // derivative = 1 - tanh²(x)
                derivative[i][j] = 1.0 - (tanhOutput[i][j] * tanhOutput[i][j]);
            }
        }

        return derivative;
    }

    // ---------------- Leaky ReLU (Optional Addition) ---------------- //
    public static double[][] leakyRelu(double[][] input, double alpha) {
        int rows = input.length, cols = input[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = input[i][j] > 0 ? input[i][j] : alpha * input[i][j];
            }
        }

        return result;
    }

    // ---------------- Leaky ReLU Derivative (Optional Addition) ---------------- //
    public static double[][] leakyReluDerivative(double[][] input, double alpha) {
        int rows = input.length, cols = input[0].length;
        double[][] derivative = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                derivative[i][j] = input[i][j] > 0 ? 1.0 : alpha;
            }
        }

        return derivative;
    }
}