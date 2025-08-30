package org.example;

public class Backward {
    private Forward forward;
    private Loss loss;

    // ===== Gradients for Weights and Biases =====
    private double[][] dW1, dW2, dW3;
    private double[][] db1, db2, db3;

    // ===== Gradients for Batch Normalization =====
    private double[][] dGamma1, dBeta1;  // Layer 1
    private double[][] dGamma2, dBeta2;  // Layer 2 (if exists)

    // ===== Setters =====
    public void setForwardAndLoss(Forward forward, Loss loss) {
        this.forward = forward;
        this.loss = loss;
    }

    // ===== Backpropagation =====
    /**
     * Compute gradients for a 3-layer neural network with dropout and batch norm
     */
    public void computeGradients(double[][] X_batch, double[][] Y_batch,
                                 double[][] W1, double[][] W2, double[][] W3,
                                 double[][] b1, double[][] b2, double[][] b3) {

        int batchSize = X_batch.length;

        // Forward pass (done once)
        forward.forward(X_batch, W1, b1, W2, b2, W3, b3);

        // Forward results
        double[][] net1 = forward.getNet1();
        double[][] out1 = forward.getOut1();
        double[][] net2 = forward.getNet2();
        double[][] out2 = forward.getOut2();
        double[][] net3 = forward.getNet3();
        double[][] out3 = forward.getOut3();

        // ---------- Output Layer ----------
        double[][] dZ3 = Matrix_Operations.subtract(out3, Y_batch);
        dZ3 = Matrix_Operations.scalarMultiply(dZ3, 1.0 / batchSize);
        dZ3 = dropoutBackward(dZ3, 0.2);

        dW3 = Matrix_Operations.multiply(Matrix_Operations.transpose(out2), dZ3);
        db3 = sumColumns(dZ3);

        // ---------- Hidden Layer 2 ----------
        double[][] dA2 = Matrix_Operations.multiply(dZ3, Matrix_Operations.transpose(W3));

        // BatchNorm (if exists)
        if (forward.getGamma2() != null && forward.getBeta2() != null) {
            dA2 = batchNormBackwardLayer2(dA2, out2, forward.getGamma2(), forward.getBeta2());
        }

        dA2 = dropoutBackward(dA2, 0.3);
        double[][] dZ2 = reluBackward(dA2, net2);

        dW2 = Matrix_Operations.multiply(Matrix_Operations.transpose(out1), dZ2);
        db2 = sumColumns(dZ2);

        // ---------- Hidden Layer 1 ----------
        double[][] dA1 = Matrix_Operations.multiply(dZ2, Matrix_Operations.transpose(W2));
        double[][] dA1_beforeBN = batchNormBackward(dA1, out1, forward.getGamma1(), forward.getBeta1());

        double[][] dZ1 = reluBackward(dA1_beforeBN, net1);

        dW1 = Matrix_Operations.multiply(Matrix_Operations.transpose(X_batch), dZ1);
        db1 = sumColumns(dZ1);
    }

    // ===== Utility: Dropout =====
    private double[][] dropoutBackward(double[][] dOut, double dropoutRate) {
        double scale = 1.0 / (1.0 - dropoutRate);
        return Matrix_Operations.scalarMultiply(dOut, scale);
    }

    // ===== Utility: ReLU =====
    private double[][] reluBackward(double[][] dA, double[][] net) {
        int rows = dA.length;
        int cols = dA[0].length;
        double[][] dZ = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dZ[i][j] = (net[i][j] > 0) ? dA[i][j] : 0.0;
            }
        }
        return dZ;
    }

    // ===== Utility: BatchNorm Layer 2 =====
    private double[][] batchNormBackwardLayer2(double[][] dOut, double[][] normalizedOutput,
                                               double[][] gamma, double[][] beta) {
        int batchSize = normalizedOutput.length;
        int features = normalizedOutput[0].length;

        dGamma2 = new double[1][features];
        dBeta2 = new double[1][features];

        for (int j = 0; j < features; j++) {
            double dBetaSum = 0.0;
            double dGammaSum = 0.0;
            for (int i = 0; i < batchSize; i++) {
                dBetaSum += dOut[i][j];
                dGammaSum += dOut[i][j] * normalizedOutput[i][j];
            }
            dBeta2[0][j] = dBetaSum;
            dGamma2[0][j] = dGammaSum;
        }

        double[][] dInput = new double[batchSize][features];
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < features; j++) {
                dInput[i][j] = gamma[0][j] * dOut[i][j];
            }
        }
        return dInput;
    }

    // ===== Utility: BatchNorm Layer 1 =====
    private double[][] batchNormBackward(double[][] dOut, double[][] normalizedOutput,
                                         double[][] gamma, double[][] beta) {
        int batchSize = normalizedOutput.length;
        int features = normalizedOutput[0].length;

        dGamma1 = new double[1][features];
        dBeta1 = new double[1][features];

        for (int j = 0; j < features; j++) {
            double dBetaSum = 0.0;
            double dGammaSum = 0.0;
            for (int i = 0; i < batchSize; i++) {
                dBetaSum += dOut[i][j];
                dGammaSum += dOut[i][j] * normalizedOutput[i][j];
            }
            dBeta1[0][j] = dBetaSum;
            dGamma1[0][j] = dGammaSum;
        }

        double[][] dInput = new double[batchSize][features];
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < features; j++) {
                dInput[i][j] = gamma[0][j] * dOut[i][j];
            }
        }
        return dInput;
    }

    // ===== Utility: Bias Gradients =====
    private double[][] sumColumns(double[][] matrix) {
        int cols = matrix[0].length;
        double[][] result = new double[1][cols];

        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int i = 0; i < matrix.length; i++) {
                sum += matrix[i][j];
            }
            result[0][j] = sum;
        }
        return result;
    }

    // ===== Update Parameters =====
    public void updateWeights(double[][] W1, double[][] W2, double[][] W3,
                              double[][] b1, double[][] b2, double[][] b3,
                              double learningRate) {

        updateMatrixInPlace(W1, dW1, learningRate);
        updateMatrixInPlace(W2, dW2, learningRate);
        updateMatrixInPlace(W3, dW3, learningRate);

        updateMatrixInPlace(b1, db1, learningRate);
        updateMatrixInPlace(b2, db2, learningRate);
        updateMatrixInPlace(b3, db3, learningRate);

        if (dGamma1 != null && dBeta1 != null) {
            updateMatrixInPlace(forward.getGamma1(), dGamma1, learningRate * 0.1);
            updateMatrixInPlace(forward.getBeta1(), dBeta1, learningRate * 0.1);
        }

        if (dGamma2 != null && dBeta2 != null &&
                forward.getGamma2() != null && forward.getBeta2() != null) {
            updateMatrixInPlace(forward.getGamma2(), dGamma2, learningRate * 0.1);
            updateMatrixInPlace(forward.getBeta2(), dBeta2, learningRate * 0.1);
        }
    }

    private void updateMatrixInPlace(double[][] matrix, double[][] gradient, double learningRate) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] -= learningRate * gradient[i][j];
            }
        }
    }

    // ===== Debugging Helpers =====
    public void printGradientNorms() {
        if (dW1 != null && dW2 != null && dW3 != null) {
            System.out.printf("Gradient norms: W1=%.6f, W2=%.6f, W3=%.6f\n",
                    computeNorm(dW1), computeNorm(dW2), computeNorm(dW3));
        }
    }

    private double computeNorm(double[][] matrix) {
        double sum = 0.0;
        for (double[] row : matrix) {
            for (double val : row) {
                sum += val * val;
            }
        }
        return Math.sqrt(sum);
    }

    // ===== Getters (For Debugging) =====
    public double[][] getDW1() { return dW1; }
    public double[][] getDW2() { return dW2; }
    public double[][] getDW3() { return dW3; }
    public double[][] getDB1() { return db1; }
    public double[][] getDB2() { return db2; }
    public double[][] getDB3() { return db3; }
    public double[][] getDGamma1() { return dGamma1; }
    public double[][] getDBeta1() { return dBeta1; }
    public double[][] getDGamma2() { return dGamma2; }
    public double[][] getDBeta2() { return dBeta2; }
}
