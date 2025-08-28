package org.example;
import java.io.*;
import java.util.Random;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {

    private static final String trainImagesPath = "data/train-images.idx3-ubyte";
    private static final String trainImageLabelsPath = "data/train-labels.idx1-ubyte";

    private static final String testImagesPath = "data/t10k-images.idx3-ubyte";
    private static final String testImageLabelsPath = "data/t10k-labels.idx1-ubyte";

    private static final int inputSize = 28*28;
    private static final int hiddenSize1 = 20;
    private static final int hiddenSize2 = 15;
    private static final int outputSize = 10;
    private static final int batchSize = 32;
    private static final int numEpochs = 10;
    private static final double learningRate = 0.01;


    private static void initWeights(double[][] matrix, Random rand) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = rand.nextGaussian() * 0.01;
            }
        }
    }

    public static void main(String[] args) throws IOException {

        Load_dataset loadDataset = new Load_dataset();

        try {
            DataSet trainData= loadDataset.loadData(trainImagesPath,trainImageLabelsPath);
            DataSet testData= loadDataset.loadData(testImagesPath,testImageLabelsPath);

            int[][] trainImages = trainData.getImages();
            int[] trainImagesLabels = trainData.getLabels();
            System.out.println("Train images: " + trainData.getSize());

            int[][] testImages = testData.getImages();
            int[] testImagesLabels = testData.getLabels();
            System.out.println("Test images: " + testData.getSize());

        }catch (IOException e){
            System.err.println("Error loading data: " + e.getMessage());
        }

        Random rand = new Random();
        double[][] W1 = new double[inputSize][hiddenSize1];
        double[] b1 = new double[hiddenSize1];

        double[][] W2 = new double[hiddenSize1][hiddenSize2];
        double[] b2 = new double[hiddenSize2];

        double[][] W3 = new double[hiddenSize2][outputSize];
        double[] b3 = new double[outputSize];

        initWeights(W1,rand);
        initWeights(W2,rand);
        initWeights(W3,rand);

















//        Loss lossCalculator = new Loss();
//        double [] groungtruth={0,1,0,0};
//        double [] prediction = {0.0,0.8,0.0,0.0};
//
//        double crossEntropyLoss=lossCalculator.calculate_loss(groungtruth,prediction);
//        System.out.println(crossEntropyLoss);
    }
}