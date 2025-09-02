package org.example;

import java.io.IOException;
import java.util.Random;

public class Main {

    // Network parameters
    private static final int INPUT_SIZE = 784;  // 28x28 pixels for MNIST
    private static final int HIDDEN1_SIZE = 128;
    private static final int HIDDEN2_SIZE = 64;
    private static final int OUTPUT_SIZE = 10;  // 10 digits (0-9)

    // Training parameters
    private static final int EPOCHS = 10;
    private static final int BATCH_SIZE = 32;
    private static final double LEARNING_RATE = 0.001;
    private static final double MOMENTUM_BETA = 0.9; // Momentum coefficient (Beta)

    // File paths for MNIST dataset
    private static final String TRAIN_IMAGES_PATH = "data/train-images.idx3-ubyte";
    private static final String TRAIN_LABELS_PATH = "data/train-labels.idx1-ubyte";
    private static final String TEST_IMAGES_PATH = "data/t10k-images.idx3-ubyte";
    private static final String TEST_LABELS_PATH = "data/t10k-labels.idx1-ubyte";

    // Model file
    private static final String MODEL_FILE = "saved_model.dat";

    public static void main(String[] args) {
        System.out.println("Starting Neural Network Program...");
        System.out.println("=".repeat(60));

        try {
            Load_dataset dataLoader = new Load_dataset();
            Forward forward = new Forward();
            Backward backward = new Backward();
            Loss lossFunction = new Loss();

            backward.setForwardAndLoss(forward, lossFunction);

            // Load datasets
            System.out.println("Loading MNIST dataset...");
            DataSet trainSet = dataLoader.loadData(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH);
            DataSet testSet = dataLoader.loadData(TEST_IMAGES_PATH, TEST_LABELS_PATH);

            // Normalize and prepare data
            double[][] X_train = normalizeData(convertToDouble(trainSet.getImages()));
            double[][] Y_train = oneHotEncode(trainSet.getLabels(), OUTPUT_SIZE);
            double[][] X_test  = normalizeData(convertToDouble(testSet.getImages()));
            double[][] Y_test  = oneHotEncode(testSet.getLabels(), OUTPUT_SIZE);

            // Check if saved model exists
            ModelIO.ModelParameters params;
            if (ModelIO.modelExists(MODEL_FILE)) {
                System.out.println("Model file found! Loading model...");
                params = ModelIO.loadModel(MODEL_FILE);
            } else {
                System.out.println("No saved model found. Starting training...");
                params = trainModel(forward, backward, lossFunction, X_train, Y_train, X_test, Y_test);
                ModelIO.saveModel(params, MODEL_FILE);
            }

            // Evaluate final model
            System.out.println("\nFinal Model Evaluation:");
            double finalTestAccuracy = evaluateModel(forward, X_test, Y_test,
                    params.W1, params.b1, params.W2, params.b2, params.W3, params.b3);
            System.out.printf("Final Test Accuracy: %.2f%%\n", finalTestAccuracy);

            // Show sample predictions
            System.out.println("\nSample Predictions:");
            testSamplePredictions(forward, X_test, testSet.getLabels(),
                    params.W1, params.b1, params.W2, params.b2, params.W3, params.b3);

        } catch (IOException e) {
            System.err.println("Error loading dataset: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Train the neural network model (SGD + Momentum)
     */
    private static ModelIO.ModelParameters trainModel(Forward forward, Backward backward, Loss lossFunction,
                                                      double[][] X_train, double[][] Y_train,
                                                      double[][] X_test,  double[][] Y_test) {
        // Initialize weights and biases
        double[][] W1 = initializeWeights(INPUT_SIZE, HIDDEN1_SIZE);
        double[][] W2 = initializeWeights(HIDDEN1_SIZE, HIDDEN2_SIZE);
        double[][] W3 = initializeWeights(HIDDEN2_SIZE, OUTPUT_SIZE);

        double[][] b1 = initializeBiases(1, HIDDEN1_SIZE);
        double[][] b2 = initializeBiases(1, HIDDEN2_SIZE);
        double[][] b3 = initializeBiases(1, OUTPUT_SIZE);

        // Initialize velocities (same shapes as weights/biases)
        double[][] vW1 = new double[INPUT_SIZE][HIDDEN1_SIZE];
        double[][] vW2 = new double[HIDDEN1_SIZE][HIDDEN2_SIZE];
        double[][] vW3 = new double[HIDDEN2_SIZE][OUTPUT_SIZE];

        double[][] vB1 = new double[1][HIDDEN1_SIZE];
        double[][] vB2 = new double[1][HIDDEN2_SIZE];
        double[][] vB3 = new double[1][OUTPUT_SIZE];

        System.out.printf("Network Architecture: %d → %d → %d → %d\n",
                INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);

        // Training loop
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            double epochLoss = 0.0;
            int correct = 0;
            int totalBatches = (int) Math.ceil((double) X_train.length / BATCH_SIZE);

            shuffleData(X_train, Y_train);

            for (int batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
                int startIdx = batchIdx * BATCH_SIZE;
                int endIdx = Math.min(startIdx + BATCH_SIZE, X_train.length);
                int currentBatchSize = endIdx - startIdx;

                double[][] X_batch = new double[currentBatchSize][INPUT_SIZE];
                double[][] Y_batch = new double[currentBatchSize][OUTPUT_SIZE];

                for (int i = 0; i < currentBatchSize; i++) {
                    X_batch[i] = X_train[startIdx + i].clone();
                    Y_batch[i] = Y_train[startIdx + i].clone();
                }

                // Forward
                double[][] predictions = forward.forward(X_batch, W1, b1, W2, b2, W3, b3);

                // Loss + accuracy
                epochLoss += calculateBatchLoss(predictions, Y_batch, lossFunction);
                correct   += calculateCorrectPredictions(predictions, Y_batch);

                // Backward (compute gradients)
                backward.computeGradients(X_batch, Y_batch, W1, W2, W3, b1, b2, b3);

                // Momentum updates
                backward.updateVelocity(vW1, vW2, vW3, vB1, vB2, vB3, MOMENTUM_BETA);
                backward.updateWeights(W1, W2, W3, b1, b2, b3, vW1, vW2, vW3, vB1, vB2, vB3, LEARNING_RATE);
            }

            double avgLoss = epochLoss / totalBatches;
            double trainAccuracy = (double) correct / X_train.length * 100;

            // Evaluate on test set every 5 epochs (and last epoch)
            double testAccuracy = 0.0;
            if ((epoch + 1) % 5 == 0 || epoch == EPOCHS - 1) {
                testAccuracy = evaluateModel(forward, X_test, Y_test, W1, b1, W2, b2, W3, b3);
            }

            if ((epoch + 1) % 5 == 0 || epoch == EPOCHS - 1) {
                System.out.printf("Epoch %3d/%d | Loss: %.4f | Train Acc: %6.2f%% | Test Acc: %6.2f%%\n",
                        epoch + 1, EPOCHS, avgLoss, trainAccuracy, testAccuracy);
            } else {
                System.out.printf("Epoch %3d/%d | Loss: %.4f | Train Acc: %6.2f%%\n",
                        epoch + 1, EPOCHS, avgLoss, trainAccuracy);
            }
        }

        System.out.println("Training completed!");
        return new ModelIO.ModelParameters(W1, b1, W2, b2, W3, b3);
    }

    // ==== Helper methods ====

    private static double[][] initializeWeights(int rows, int cols) {
        double[][] weights = new double[rows][cols];
        Random random = new Random();
        double std = Math.sqrt(2.0 / rows); // He initialization
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                weights[i][j] = random.nextGaussian() * std;
            }
        }
        return weights;
    }

    private static double[][] initializeBiases(int rows, int cols) {
        return new double[rows][cols]; // zeros
    }

    private static double[][] convertToDouble(int[][] intArray) {
        double[][] doubleArray = new double[intArray.length][intArray[0].length];
        for (int i = 0; i < intArray.length; i++) {
            for (int j = 0; j < intArray[0].length; j++) {
                doubleArray[i][j] = intArray[i][j];
            }
        }
        return doubleArray;
    }

    private static double[][] normalizeData(double[][] data) {
        double[][] normalized = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                normalized[i][j] = data[i][j] / 255.0; // [0,1]
            }
        }
        return normalized;
    }

    private static double[][] oneHotEncode(int[] labels, int numClasses) {
        double[][] encoded = new double[labels.length][numClasses];
        for (int i = 0; i < labels.length; i++) {
            encoded[i][labels[i]] = 1.0;
        }
        return encoded;
    }

    private static void shuffleData(double[][] X, double[][] Y) {
        Random random = new Random();
        for (int i = X.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            double[] tempX = X[i];
            X[i] = X[j];
            X[j] = tempX;

            double[] tempY = Y[i];
            Y[i] = Y[j];
            Y[j] = tempY;
        }
    }

    private static double calculateBatchLoss(double[][] predictions, double[][] labels, Loss lossFunction) {
        double totalLoss = 0.0;
        double[][] oneHotBatch = toOneHotBatch(predictions);
        for (int i = 0; i < predictions.length; i++) {
            totalLoss += lossFunction.calculate_loss(labels[i], oneHotBatch[i]);
        }
        return totalLoss / predictions.length;
    }

    private static double[][] toOneHotBatch(double[][] predictions) {
        double[][] oneHotBatch = new double[predictions.length][];
        for (int i = 0; i < predictions.length; i++) {
            int maxIndex = argmax(predictions[i]);
            double[] oneHot = new double[predictions[i].length];
            oneHot[maxIndex] = 1.0;
            oneHotBatch[i] = oneHot;
        }
        return oneHotBatch;
    }

    private static int calculateCorrectPredictions(double[][] predictions, double[][] labels) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            int predictedClass = argmax(predictions[i]);
            int actualClass = argmax(labels[i]);
            if (predictedClass == actualClass) correct++;
        }
        return correct;
    }

    private static int argmax(double[] array) {
        int maxIndex = 0;
        double maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private static double evaluateModel(Forward forward, double[][] X_test, double[][] Y_test,
                                        double[][] W1, double[][] b1, double[][] W2, double[][] b2,
                                        double[][] W3, double[][] b3) {
        double[][] predictions = forward.forward(X_test, W1, b1, W2, b2, W3, b3);
        int correct = calculateCorrectPredictions(predictions, Y_test);
        return (double) correct / X_test.length * 100;
    }

    private static void testSamplePredictions(Forward forward, double[][] X_test, int[] actualLabels,
                                              double[][] W1, double[][] b1, double[][] W2, double[][] b2,
                                              double[][] W3, double[][] b3) {
        Random random = new Random();
        for (int i = 0; i < 5; i++) {
            int sampleIdx = random.nextInt(X_test.length);
            double[][] sampleInput = { X_test[sampleIdx] };

            double[][] prediction = forward.forward(sampleInput, W1, b1, W2, b2, W3, b3);

            int predictedClass = argmax(prediction[0]);
            int actualClass = actualLabels[sampleIdx];
            double confidence = prediction[0][predictedClass] * 100;

            String status = (predictedClass == actualClass) ? "Ok" : "No";
            System.out.printf("%s Sample %d: Predicted = %d, Actual = %d, Confidence = %.1f%%\n",
                    status, i + 1, predictedClass, actualClass, confidence);
        }
    }
}
