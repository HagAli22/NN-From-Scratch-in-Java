package org.example;

public class Forward {
    // input -> X_batch, W1, b1, W2, b2, W3, b3
    // output -> net1, out1, net2, out2, net3, out3
    
    public double[][] net1, out1;
    public double[][] net2, out2;  
    public double[][] net3, out3;
    
    public void forward(double[][] X_batch, 
                       double[][] W1, double[][] b1,
                       double[][] W2, double[][] b2, 
                       double[][] W3, double[][] b3) {
        
        Matrix_Operations matOps = new Matrix_Operations();
        Activation_Function activation = new Activation_Function();
        // Layer 1: Input -> Hidden1
        net1 = matOps.add(matOps.multiply(X_batch, W1), b1);
        out1 = activation.relu(net1);
        
        // Layer 2: Hidden1 -> Hidden2
        net2 = matOps.add(matOps.multiply(out1, W2), b2);
        out2 = activation.relu(net2);
        
        // Layer 3: Hidden2 -> Output
        net3 = matOps.add(matOps.multiply(out2, W3), b3);
        out3 = activation.softmax(net3);

    }
}