package org.example;

public class Backward {
    // ======================== GRADIENT STORAGE ========================
    private double[][] dW1, db1;  // Gradients for Layer 1
    private double[][] dW2, db2;  // Gradients for Layer 2
    private double[][] dW3, db3;  // Gradients for Layer 3

    // Batch Norm Gradients
    private double[][] dGamma1, dBeta1;

    // Regularization parameters
    private double l2Lambda = 0.0;
    private double l1Lambda = 0.0;

    /**
     * Perform full backward propagation
     */
    public void backward(double[][] X_batch, double[][] Y_true, double[][] Y_pred,
                         Forward forward,
                         double[][] W1, double[][] b1,
                         double[][] W2, double[][] b2,
                         double[][] W3, double[][] b3,
                         double learningRate) {

        int batchSize = X_batch.length;

        // ======================== LAYER 3 (OUTPUT) ========================
        double[][] dOut3 = backwardOutputLayer(Y_true, Y_pred, forward.getOut2(), W3, batchSize);

        // ======================== LAYER 2 (HIDDEN) ========================
        double[][] dOut2 = backwardHiddenLayer2(dOut3, forward.getNet2(), forward.getOut1(), W2);

        // ======================== LAYER 1 (HIDDEN) ========================
        backwardHiddenLayer1(dOut2, X_batch, forward.getNet1(), forward.getOut1(),
                forward.getGamma1(), forward.getBeta1(), W1);

        // ======================== PARAMETER UPDATES ========================
        updateParameters(W1, b1, W2, b2, W3, b3,
                forward.getGamma1(), forward.getBeta1(), learningRate);
    }

    // ======================== LAYER BACKWARD METHODS ========================

    /** Layer 3 (Softmax + CrossEntropy) */
    private double[][] backwardOutputLayer(double[][] Y_true, double[][] Y_pred,
                                           double[][] out2, double[][] W3, int batchSize) {

        // Loss gradient w.r.t. predictions
        double[][] dLoss_dY = Matrix_Operations.subtract(Y_pred, Y_true);
        dLoss_dY = Matrix_Operations.scalarMultiply(dLoss_dY, 1.0 / batchSize);

        // Dropout backward
        double[][] dOut3 = dropoutBackward(dLoss_dY, 0.2);

        // Gradients
        dW3 = Matrix_Operations.multiply(Matrix_Operations.transpose(out2), dOut3);
        db3 = sumAlongAxis0(dOut3);

        // Propagate to previous layer
        double[][] dOut2 = Matrix_Operations.multiply(dOut3, Matrix_Operations.transpose(W3));

        // Add regularization
        applyL2Regularization(W3, dW3);

        return dOut2;
    }

    /** Layer 2 (ReLU + Dropout) */
    private double[][] backwardHiddenLayer2(double[][] dOut2, double[][] net2,
                                            double[][] out1, double[][] W2) {

        // Dropout backward
        double[][] dOut2_drop = dropoutBackward(dOut2, 0.3);

        // ReLU derivative
        double[][] reluDeriv2 = reluDerivative(net2);
        double[][] dNet2 = elementwiseMultiply(dOut2_drop, reluDeriv2);

        // Gradients
        dW2 = Matrix_Operations.multiply(Matrix_Operations.transpose(out1), dNet2);
        db2 = sumAlongAxis0(dNet2);

        // Propagate to previous layer
        double[][] dOut1 = Matrix_Operations.multiply(dNet2, Matrix_Operations.transpose(W2));

        // Add regularization
        applyL2Regularization(W2, dW2);

        return dOut1;
    }

    /** Layer 1 (BatchNorm + ReLU) */
    private void backwardHiddenLayer1(double[][] dOut1, double[][] X_batch,
                                      double[][] net1, double[][] out1,
                                      double[][] gamma1, double[][] beta1,
                                      double[][] W1) {

        // BatchNorm backward
        double[][] dOut1_BN = batchNormBackward(dOut1, out1, gamma1, beta1);

        // ReLU backward
        double[][] reluDeriv1 = reluDerivative(net1);
        double[][] dNet1 = elementwiseMultiply(dOut1_BN, reluDeriv1);

        // Gradients
        dW1 = Matrix_Operations.multiply(Matrix_Operations.transpose(X_batch), dNet1);
        db1 = sumAlongAxis0(dNet1);

        // Add regularization
        applyL2Regularization(W1, dW1);
    }

    // ======================== HELPER FUNCTIONS ========================

    /** BatchNorm Backward */
    private double[][] batchNormBackward(double[][] dOut, double[][] normInput,
                                         double[][] gamma, double[][] beta) {
        int batchSize = normInput.length, features = normInput[0].length;

        dGamma1 = new double[1][features];
        dBeta1 = new double[1][features];

        // dBeta = Σ dOut
        for (int j = 0; j < features; j++) {
            for (int i = 0; i < batchSize; i++) {
                dBeta1[0][j] += dOut[i][j];
                dGamma1[0][j] += dOut[i][j] * normInput[i][j];
            }
        }

        // Approximate input gradient
        double[][] dInput = new double[batchSize][features];
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < features; j++) {
                dInput[i][j] = gamma[0][j] * dOut[i][j];
            }
        }
        return dInput;
    }

    /** Dropout backward */
    private double[][] dropoutBackward(double[][] dOut, double dropoutRate) {
        double scale = 1.0 / (1.0 - dropoutRate);
        return Matrix_Operations.scalarMultiply(dOut, scale);
    }

    /** ReLU derivative */
    private double[][] reluDerivative(double[][] net) {
        int rows = net.length, cols = net[0].length;
        double[][] deriv = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                deriv[i][j] = net[i][j] > 0 ? 1.0 : 0.0;
        return deriv;
    }

    /** Elementwise multiply */
    private double[][] elementwiseMultiply(double[][] a, double[][] b) {
        int rows = a.length, cols = a[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i][j] = a[i][j] * b[i][j];
        return result;
    }

    /** Column-wise sum */
    private double[][] sumAlongAxis0(double[][] input) {
        int cols = input[0].length;
        double[][] result = new double[1][cols];
        for (int j = 0; j < cols; j++)
            for (int i = 0; i < input.length; i++)
                result[0][j] += input[i][j];
        return result;
    }

    /** Apply L2 regularization */
    private void applyL2Regularization(double[][] W, double[][] dW) {
        if (l2Lambda > 0) {
            double[][] regGrad = Techniques.weightDecayGradient(W, l2Lambda);
            dW = Matrix_Operations.add(dW, regGrad);
        }
    }

    /** Update all parameters */
    private void updateParameters(double[][] W1, double[][] b1,
                                  double[][] W2, double[][] b2,
                                  double[][] W3, double[][] b3,
                                  double[][] gamma1, double[][] beta1,
                                  double learningRate) {

        // Weights
        updateWeights(W1, dW1, learningRate);
        updateWeights(W2, dW2, learningRate);
        updateWeights(W3, dW3, learningRate);

        // Biases
        updateWeights(b1, db1, learningRate);
        updateWeights(b2, db2, learningRate);
        updateWeights(b3, db3, learningRate);

        // BatchNorm params
        if (dGamma1 != null && dBeta1 != null) {
            updateWeights(gamma1, dGamma1, learningRate);
            updateWeights(beta1, dBeta1, learningRate);
        }
    }

    /** Gradient descent update */
    private void updateWeights(double[][] weights, double[][] gradients, double learningRate) {
        for (int i = 0; i < weights.length; i++)
            for (int j = 0; j < weights[0].length; j++)
                weights[i][j] -= learningRate * gradients[i][j];
    }

    // ======================== GETTERS ========================
    public double[][] getDW1() { return dW1; }
    public double[][] getDB1() { return db1; }
    public double[][] getDW2() { return dW2; }
    public double[][] getDB2() { return db2; }
    public double[][] getDW3() { return dW3; }
    public double[][] getDB3() { return db3; }
    public double[][] getDGamma1() { return dGamma1; }
    public double[][] getDBeta1() { return dBeta1; }

    // ======================== SETTERS ========================
    public void setL2Lambda(double lambda) { this.l2Lambda = lambda; }
    public void setL1Lambda(double lambda) { this.l1Lambda = lambda; }
}
