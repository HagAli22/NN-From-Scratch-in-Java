package org.example;

public class Backward {
    private Forward forward;
    private Loss loss;

    // Gradients for weights and biases
    public double[][] dW1, dW2, dW3;
    public double[][] db1, db2, db3;

    // Gradients for batch normalization
    public double[][] dGamma1, dBeta1;

    public void setForwardAndLoss(Forward forward, Loss loss) {
        this.forward = forward;
        this.loss = loss;
    }

    /**
     * Compute gradients for a 3-layer neural network with dropout and batch norm
     */
    public void computeGradients(double[][] X_batch, double[][] Y_batch,
                                 double[][] W1, double[][] W2, double[][] W3,
                                 double[][] b1, double[][] b2, double[][] b3) {

        int m = X_batch.length; // number of examples

        // ===== Forward pass outputs =====
        double[][] out3 = forward.forward(X_batch, W1, b1, W2, b2, W3, b3);
        double[][] out2 = forward.getOut2();
        double[][] out1 = forward.getOut1();
        double[][] net2 = forward.getNet2();
        double[][] net1 = forward.getNet1();

        // ================= Output Layer =================
        double[][] dZ3 = new double[m][Y_batch[0].length];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < Y_batch[0].length; j++)
                dZ3[i][j] = out3[i][j] - Y_batch[i][j]; // dZ = out - Y

        dW3 = Matrix_Operations.multiply(Matrix_Operations.transpose(out2), dZ3);
        db3 = new double[1][dZ3[0].length];
        for (int j = 0; j < dZ3[0].length; j++) {
            double sum = 0;
            for (int i = 0; i < m; i++) sum += dZ3[i][j];
            db3[0][j] = sum;
        }

        // ================= Hidden Layer 2 (with Batch Norm) =================
        double[][] dA2 = Matrix_Operations.multiply(dZ3, Matrix_Operations.transpose(W3));

        // Backprop through batch normalization
        double[][] dA2_bn = Techniques.batchNormBackward(dA2, out2,
                forward.getGamma1(), forward.getBeta1());

        // Store batch norm gradients
        dGamma1 = Techniques.computeGammaGradient(dA2, out2);
        dBeta1 = Techniques.computeBetaGradient(dA2);

        double[][] dZ2 = new double[dA2_bn.length][dA2_bn[0].length];
        double[][] reluDer2 = Activation_Function.reluDerivativeFromNet(net2);
        for (int i = 0; i < dA2_bn.length; i++)
            for (int j = 0; j < dA2_bn[0].length; j++)
                dZ2[i][j] = dA2_bn[i][j] * reluDer2[i][j];

        dW2 = Matrix_Operations.multiply(Matrix_Operations.transpose(out1), dZ2);
        db2 = new double[1][dZ2[0].length];
        for (int j = 0; j < dZ2[0].length; j++) {
            double sum = 0;
            for (int i = 0; i < m; i++) sum += dZ2[i][j];
            db2[0][j] = sum;
        }

        // ================= Hidden Layer 1 (with Dropout) =================
        double[][] dA1 = Matrix_Operations.multiply(dZ2, Matrix_Operations.transpose(W2));

        // Backprop through dropout (assuming dropout mask is saved in Techniques)
        dA1 = Techniques.dropoutBackward(dA1, 0.3);

        double[][] dZ1 = new double[dA1.length][dA1[0].length];
        double[][] reluDer1 = Activation_Function.reluDerivativeFromNet(net1);
        for (int i = 0; i < dA1.length; i++)
            for (int j = 0; j < dA1[0].length; j++)
                dZ1[i][j] = dA1[i][j] * reluDer1[i][j];

        dW1 = Matrix_Operations.multiply(Matrix_Operations.transpose(X_batch), dZ1);
        db1 = new double[1][dZ1[0].length];
        for (int j = 0; j < dZ1[0].length; j++) {
            double sum = 0;
            for (int i = 0; i < m; i++) sum += dZ1[i][j];
            db1[0][j] = sum;
        }
    }

    /**
     * Apply gradient descent update - Fixed version
     */
    public void updateWeights(double[][] W1, double[][] W2, double[][] W3,
                              double[][] b1, double[][] b2, double[][] b3,
                              double learningRate) {

        // Update weights in-place
        updateMatrixInPlace(W1, dW1, learningRate);
        updateMatrixInPlace(W2, dW2, learningRate);
        updateMatrixInPlace(W3, dW3, learningRate);

        updateMatrixInPlace(b1, db1, learningRate);
        updateMatrixInPlace(b2, db2, learningRate);
        updateMatrixInPlace(b3, db3, learningRate);

        // Update batch normalization parameters
        if (dGamma1 != null && dBeta1 != null) {
            updateMatrixInPlace(forward.getGamma1(), dGamma1, learningRate);
            updateMatrixInPlace(forward.getBeta1(), dBeta1, learningRate);
        }
    }

    /**
     * Helper method to update matrix in-place
     */
    private void updateMatrixInPlace(double[][] matrix, double[][] gradient, double learningRate) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] -= learningRate * gradient[i][j];
            }
        }
    }
}