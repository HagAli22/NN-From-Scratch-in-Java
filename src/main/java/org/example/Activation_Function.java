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



}
