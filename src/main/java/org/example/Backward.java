package org.example;

import java.util.HashMap;

public class Backward {
    private Forward forward;
    private Loss loss;

    // Gradients for weights and biases
    public double[][] dW1, dW2, dW3;
    public double[][] db1, db2, db3;


    public void setForwardAndLoss(Forward forward, Loss loss) {
        this.forward = forward;
        this.loss = loss;
    }

    /**
     * Compute gradients for a 3-layer neural network
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

        double[][] dZ2 = new double[dA2.length][dA2[0].length];
        double[][] reluDer2 = Activation_Function.reluDerivativeFromNet(net2);
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

        // ================= Hidden Layer 1 (with Dropout) =================
        double[][] dA1 = Matrix_Operations.multiply(dZ2, Matrix_Operations.transpose(W2));


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


    public void updateVelocity(double[][] velocityW1, double[][] velocityW2, double[][] velocityW3,
                              double[][] velocityB1, double[][] velocityB2, double[][] velocityB3,
                              double Beta) {

        // Update velocity in-place
        updateVelocityInPlace(velocityW1, dW1, Beta);
        updateVelocityInPlace(velocityW2, dW2, Beta);
        updateVelocityInPlace(velocityW3, dW3, Beta);

        updateVelocityInPlace(velocityB1, db1, Beta);
        updateVelocityInPlace(velocityB2, db2, Beta);
        updateVelocityInPlace(velocityB3, db3, Beta);


    }


    private void updateVelocityInPlace(double[][] matrixVelocity, double[][] gradient, double Beta) {
        for (int i = 0; i < matrixVelocity.length; i++) {
            for (int j = 0; j < matrixVelocity[0].length; j++) {
                matrixVelocity[i][j] = Beta * matrixVelocity[i][j] + gradient[i][j];
            }
        }
    }


    public void updateWeights(double[][] W1, double[][] W2, double[][] W3,
                              double[][] b1, double[][] b2, double[][] b3,
                              double[][] velocityW1, double[][] velocityW2, double[][] velocityW3,
                              double[][] velocityB1, double[][] velocityB2, double[][] velocityB3,
                              double learningRate) {

        // Update weights in-place
        updateWeightsInPlace(W1, velocityW1, learningRate);
        updateWeightsInPlace(W2, velocityW2, learningRate);
        updateWeightsInPlace(W3, velocityW3, learningRate);

        updateWeightsInPlace(b1, velocityB1, learningRate);
        updateWeightsInPlace(b2, velocityB2, learningRate);
        updateWeightsInPlace(b3, velocityB3, learningRate);


    }

    private void updateWeightsInPlace(double[][] matrixWeights, double[][] matrixVelocity, double learningRate) {
        for (int i = 0; i < matrixWeights.length; i++) {
            for (int j = 0; j < matrixWeights[0].length; j++) {
                matrixWeights[i][j] -= learningRate * matrixVelocity[i][j];
            }
        }
    }
}