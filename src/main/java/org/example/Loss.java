package org.example;
import java.util.*;
public class Loss {

    private static final double EPSILON = 1e-15;

    private void validateInput(double[]groundTruth , double[]predictions){

        if (groundTruth == null || predictions == null){
            throw new IllegalArgumentException("Input arrays cannot be null");
        }

        if (groundTruth.length != predictions.length){
            throw new IllegalArgumentException("Ground Truth and predictions must have same length");
        }

        if (groundTruth.length == 0){
            throw new IllegalArgumentException("Ground Truth cannot be empty");
        }
    }

    private double [] computeLogPredictions(double[] predictions){

        double [] logOfPredictions = new double[predictions.length];

        for (int i = 0; i < predictions.length; i++){

            predictions[i] += EPSILON;

            logOfPredictions[i]=Math.log(predictions[i]);

        }

        return logOfPredictions;
    }

    private double [] multiplyElementwise(double[] groundTruth, double [] LogOfPredictions){

        double[] elementwiseProduct = new double[groundTruth.length];

        for (int i = 0 ; i < groundTruth.length; i++){

            elementwiseProduct[i] = groundTruth[i] * LogOfPredictions[i] ;

        }

        return elementwiseProduct;

    }

    private double computeNegativeSum(double[] values){

        double sum=0.0;

        for (int i = 0; i < values.length; i++){

            sum += values[i];

        }

        return -sum;

    }
    public double calculate_loss(double[] groundTruth, double[] predictions){

        validateInput(groundTruth,predictions);

        double[] logOfPredictions=computeLogPredictions(predictions);

        double[] elementwiseProduct = multiplyElementwise(groundTruth,logOfPredictions);

        return computeNegativeSum(elementwiseProduct);

    }
}