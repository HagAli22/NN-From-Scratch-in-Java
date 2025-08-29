package org.example;

import java.util.Random;

public class Techniques {

    private static Random random = new Random();



    // ======================= DROPOUT =======================


    public static double[][] dropout(double[][] input, double dropoutRate, boolean isTraining) {
        if (!isTraining || dropoutRate <= 0.0) {
            return input; // No dropout during inference or if rate is 0
        }

        int rows = input.length;
        int cols = input[0].length;
        double[][] result = new double[rows][cols];

        // Scale factor to maintain expected output during training
        double scale = 1.0 / (1.0 - dropoutRate);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (random.nextDouble() > dropoutRate) {
                    result[i][j] = input[i][j] * scale;
                } else {
                    result[i][j] = 0.0;
                }
            }
        }

        return result;
    }



}