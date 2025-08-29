package org.example;

public class Forward {
    // input -> X_batch, W1, b1, W2, b2, W3, b3
    // output -> net1, out1, net2, out2, net3, out3

    double[][] net1;
    double[][] out1;
    double[][] net2;
    double[][] out2;
    private double[][] net3, out3;

    public double[][] forward(double[][] X_batch,
                              double[][] W1, double[][] b1,
                              double[][] W2, double[][] b2,
                              double[][] W3, double[][] b3) {

        // Layer 1: Input -> Hidden1
        net1 = Matrix_Operations.add(Matrix_Operations.multiply(X_batch, W1), b1);
        out1 = Activation_Function.relu(net1);

        // Layer 2: Hidden1 -> Hidden2
        net2 = Matrix_Operations.add(Matrix_Operations.multiply(out1, W2), b2);
        out2 = Activation_Function.relu(net2);

        // Layer 3: Hidden2 -> Output
        net3 = Matrix_Operations.add(Matrix_Operations.multiply(out2, W3), b3);
        out3 = Activation_Function.softmax(net3);

        return out3;
    }

    /*
    public double[][] getNet1() { return net1; }
    public double[][] getOut1() { return out1; }
    public double[][] getNet2() { return net2; }
    public double[][] getOut2() { return out2; }
    public double[][] getNet3() { return net3; }
    public double[][] getOut3() { return out3; }
    */
}
