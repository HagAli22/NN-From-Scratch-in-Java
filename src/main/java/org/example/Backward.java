package org.example;

public class Backward {
    private Forward forward;
    private Loss loss;

    // Gradients
    public double[][] dW1, dW2, dW3;
    public double[][] db1, db2, db3;

    public void setForwardAndLoss(Forward forward, Loss loss) {
        this.forward = forward;
        this.loss = loss;
    }

    /**
     * Compute gradients for a 3-layer neural network
     * @param X_batch  Input data [m x input_size]
     * @param Y_batch  One-hot labels [m x output_size]
     * @param W1, W2, W3  Current weights
     * @param b1, b2, b3  Current biases
     */
    public void computeGradients(double[][] X_batch, double[][] Y_batch,
                                 double[][] W1, double[][] W2, double[][] W3,
                                 double[][] b1, double[][] b2, double[][] b3) {

        int m = X_batch.length; // number of examples

        // ===== Forward pass outputs =====
        double[][] out3 = forward.forward(X_batch, W1, b1, W2, b2, W3, b3);
        double[][] out2 = forward.out2;
        double[][] out1 = forward.out1;
        double[][] net2 = forward.net2;
        double[][] net1 = forward.net1;

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

        // ================= Hidden Layer 2 =================
        double[][] dA2 = Matrix_Operations.multiply(dZ3, Matrix_Operations.transpose(W3));
        double[][] dZ2 = new double[dA2.length][dA2[0].length];
        double[][] reluDer2 = Activation_Function.reluDerivativeFromOutput(out2);
        for (int i = 0; i < dA2.length; i++)
            for (int j = 0; j < dA2[0].length; j++)
                dZ2[i][j] = dA2[i][j] * reluDer2[i][j];

        dW2 = Matrix_Operations.multiply(Matrix_Operations.transpose(out1), dZ2);
        db2 = new double[1][dZ2[0].length];
        for (int j = 0; j < dZ2[0].length; j++) {
            double sum = 0;
            for (int i = 0; i < m; i++) sum += dZ2[i][j];
            db2[0][j] = sum;
        }

        // ================= Hidden Layer 1 =================
        double[][] dA1 = Matrix_Operations.multiply(dZ2, Matrix_Operations.transpose(W2));
        double[][] dZ1 = new double[dA1.length][dA1[0].length];
        double[][] reluDer1 = Activation_Function.reluDerivativeFromOutput(out1);
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
     * Apply gradient descent update
     */
    public void updateWeights(double[][] W1, double[][] W2, double[][] W3,
                              double[][] b1, double[][] b2, double[][] b3,
                              double learningRate) {

        W1 = Matrix_Operations.subtract(W1, Matrix_Operations.scalarMultiply(dW1, learningRate));
        W2 = Matrix_Operations.subtract(W2, Matrix_Operations.scalarMultiply(dW2, learningRate));
        W3 = Matrix_Operations.subtract(W3, Matrix_Operations.scalarMultiply(dW3, learningRate));

        b1 = Matrix_Operations.subtract(b1, Matrix_Operations.scalarMultiply(db1, learningRate));
        b2 = Matrix_Operations.subtract(b2, Matrix_Operations.scalarMultiply(db2, learningRate));
        b3 = Matrix_Operations.subtract(b3, Matrix_Operations.scalarMultiply(db3, learningRate));
    }
}
