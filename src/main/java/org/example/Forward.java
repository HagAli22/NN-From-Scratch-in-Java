package org.example;

public class Forward {
    // input -> X_batch, W1, b1, W2, b2, W3, b3
    // output -> net1, out1, net2, out2, net3, out3

    double[][] net1;
    double[][] out1;
    double[][] net2;
    double[][] out2;
    private double[][] net3, out3;

    // Batch normalization parameters
    private double[][] gamma1, beta1; // For layer 1
    private double[][] gamma2, beta2; // For layer 2

    public double[][] forward(double[][] X_batch,
                              double[][] W1, double[][] b1,
                              double[][] W2, double[][] b2,
                              double[][] W3, double[][] b3) {

        // Layer 1: Input -> Hidden1
        net1 = Matrix_Operations.add(Matrix_Operations.multiply(X_batch, W1), b1);
        out1 = Activation_Function.relu(net1);
        out1 = Techniques.dropout(out1, 0.3, true);


        // Layer 2: Hidden1 -> Hidden2
        net2 = Matrix_Operations.add(Matrix_Operations.multiply(out1, W2), b2);
        out2 = Activation_Function.relu(net2);


        // Initialize gamma/beta if using batch norm for first time
        if (gamma1 == null) {
            gamma1 = Techniques.initializeGamma(out1[0].length);
            beta1 = Techniques.initializeBeta(out1[0].length);
        }
        out2 = Techniques.batchNormalization(out2, gamma1, beta1);


        // Layer 3: Hidden2 -> Output
        net3 = Matrix_Operations.add(Matrix_Operations.multiply(out2, W3), b3);
        out3 = Activation_Function.softmax(net3);


        return out3;
    }

    // Getters for backpropagation
    public double[][] getNet1() { return net1; }
    public double[][] getOut1() { return out1; }
    public double[][] getNet2() { return net2; }
    public double[][] getOut2() { return out2; }
    public double[][] getNet3() { return net3; }
    public double[][] getOut3() { return out3; }

    // Getters for batch norm parameters
    public double[][] getGamma1() { return gamma1; }
    public double[][] getBeta1() { return beta1; }
    public double[][] getGamma2() { return gamma2; }
    public double[][] getBeta2() { return beta2; }
}