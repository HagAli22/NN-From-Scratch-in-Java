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
    private static final int EPOCHS = 50;
    private static final int BATCH_SIZE = 32;
    private static final double LEARNING_RATE = 0.001;
    private static final double L2_LAMBDA = 0.001;

    // File paths for MNIST dataset
    private static final String TRAIN_IMAGES_PATH = "D:\\NN-From-Scratch-in-Java\\data\\train-images.idx3-ubyte";
    private static final String TRAIN_LABELS_PATH = "D:\\NN-From-Scratch-in-Java\\data\\train-labels.idx1-ubyte";
    private static final String TEST_IMAGES_PATH = "D:\\NN-From-Scratch-in-Java\\data\\t10k-images.idx3-ubyte";
    private static final String TEST_LABELS_PATH = "D:\\NN-From-Scratch-in-Java\\data\\t10k-labels.idx1-ubyte";

    public static void main(String[] args) {
        System.out.println("🚀 Starting Neural Network Training...");
        System.out.println("=" .repeat(60));

        try {
            // Initialize components
            Load_dataset dataLoader = new Load_dataset();
            Forward forward = new Forward();
            Backward backward = new Backward();
            Loss lossFunction = new Loss();

            // Set forward and loss in backward
            backward.setForwardAndLoss(forward, lossFunction);

            // Load datasets
            System.out.println("📂 Loading MNIST dataset...");
            DataSet trainSet = dataLoader.loadData(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH);
            DataSet testSet = dataLoader.loadData(TEST_IMAGES_PATH, TEST_LABELS_PATH);

            System.out.printf("✅ Training set: %d samples\n", trainSet.getSize());
            System.out.printf("✅ Test set: %d samples\n", testSet.getSize());

            // Initialize network weights and biases
            System.out.println("\n🔧 Initializing network parameters...");
            double[][] W1 = initializeWeights(INPUT_SIZE, HIDDEN1_SIZE);
            double[][] W2 = initializeWeights(HIDDEN1_SIZE, HIDDEN2_SIZE);
            double[][] W3 = initializeWeights(HIDDEN2_SIZE, OUTPUT_SIZE);

            double[][] b1 = initializeBiases(1, HIDDEN1_SIZE);
            double[][] b2 = initializeBiases(1, HIDDEN2_SIZE);
            double[][] b3 = initializeBiases(1, OUTPUT_SIZE);

            // Convert training data
            double[][] X_train = normalizeData(convertToDouble(trainSet.getImages()));
            double[][] Y_train = oneHotEncode(trainSet.getLabels(), OUTPUT_SIZE);

            double[][] X_test = normalizeData(convertToDouble(testSet.getImages()));
            double[][] Y_test = oneHotEncode(testSet.getLabels(), OUTPUT_SIZE);

            System.out.printf("🎯 Network Architecture: %d → %d → %d → %d\n",
                    INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);
            System.out.println("\n🏋️ Starting training...");
            System.out.println("=" .repeat(60));

            // Training loop
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                double epochLoss = 0.0;
                int correct = 0;
                int totalBatches = (int) Math.ceil((double) X_train.length / BATCH_SIZE);

                // Shuffle data for each epoch
                shuffleData(X_train, Y_train);

                for (int batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
                    // Clear cache before each batch
                    Techniques.clearCache();

                    // Create batch
                    int startIdx = batchIdx * BATCH_SIZE;
                    int endIdx = Math.min(startIdx + BATCH_SIZE, X_train.length);
                    int currentBatchSize = endIdx - startIdx;

                    double[][] X_batch = new double[currentBatchSize][INPUT_SIZE];
                    double[][] Y_batch = new double[currentBatchSize][OUTPUT_SIZE];

                    for (int i = 0; i < currentBatchSize; i++) {
                        X_batch[i] = X_train[startIdx + i].clone();
                        Y_batch[i] = Y_train[startIdx + i].clone();
                    }

                    // Forward pass
                    double[][] predictions = forward.forward(X_batch, W1, b1, W2, b2, W3, b3);

                    // Calculate batch loss
                    double batchLoss = calculateBatchLoss(predictions, Y_batch, lossFunction);

                    // Add regularization
                    double regLoss = Techniques.l2Regularization(W1, L2_LAMBDA) +
                            Techniques.l2Regularization(W2, L2_LAMBDA) +
                            Techniques.l2Regularization(W3, L2_LAMBDA);
                    batchLoss += regLoss;

                    epochLoss += batchLoss;

                    // Calculate accuracy
                    correct += calculateCorrectPredictions(predictions, Y_batch);

                    // Backward pass
                    backward.computeGradients(X_batch, Y_batch, W1, W2, W3, b1, b2, b3);

                    // Add regularization to gradients
                    addRegularizationGradients(backward, W1, W2, W3, L2_LAMBDA);

                    // Update weights
                    backward.updateWeights(W1, W2, W3, b1, b2, b3, LEARNING_RATE);
                }

                // Calculate epoch metrics
                double avgLoss = epochLoss / totalBatches;
                double trainAccuracy = (double) correct / X_train.length * 100;

                // Test accuracy every 5 epochs
                double testAccuracy = 0.0;
                if ((epoch + 1) % 5 == 0 || epoch == EPOCHS - 1) {
                    testAccuracy = evaluateModel(forward, X_test, Y_test, W1, b1, W2, b2, W3, b3);
                }

                // Print progress
                if ((epoch + 1) % 5 == 0 || epoch == EPOCHS - 1) {
                    System.out.printf("Epoch %3d/%d | Loss: %.4f | Train Acc: %6.2f%% | Test Acc: %6.2f%%\n",
                            epoch + 1, EPOCHS, avgLoss, trainAccuracy, testAccuracy);
                } else {
                    System.out.printf("Epoch %3d/%d | Loss: %.4f | Train Acc: %6.2f%%\n",
                            epoch + 1, EPOCHS, avgLoss, trainAccuracy);
                }
            }

            System.out.println("=" .repeat(60));
            System.out.println("🎉 Training completed successfully!");

            // Final evaluation
            System.out.println("\n📊 Final Model Evaluation:");
            double finalTestAccuracy = evaluateModel(forward, X_test, Y_test, W1, b1, W2, b2, W3, b3);
            System.out.printf("🎯 Final Test Accuracy: %.2f%%\n", finalTestAccuracy);

            // Test on a few samples
            System.out.println("\n🔍 Sample Predictions:");
            testSamplePredictions(forward, X_test, testSet.getLabels(), W1, b1, W2, b2, W3, b3);

        } catch (IOException e) {
            System.err.println("❌ Error loading dataset: " + e.getMessage());
            System.err.println("💡 Make sure MNIST dataset files are in the project directory:");
            System.err.println("   - " + TRAIN_IMAGES_PATH);
            System.err.println("   - " + TRAIN_LABELS_PATH);
            System.err.println("   - " + TEST_IMAGES_PATH);
            System.err.println("   - " + TEST_LABELS_PATH);
        } catch (Exception e) {
            System.err.println("❌ Unexpected error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // Helper methods

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
        return new double[rows][cols]; // Initialize to zeros
    }

    // Alternative: Create bias with batch size
    private static double[][] initializeBiasesForBatch(int batchSize, int cols) {
        return new double[batchSize][cols]; // Initialize to zeros
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
                normalized[i][j] = data[i][j] / 255.0; // Normalize to [0,1]
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

            // Swap X
            double[] tempX = X[i];
            X[i] = X[j];
            X[j] = tempX;

            // Swap Y
            double[] tempY = Y[i];
            Y[i] = Y[j];
            Y[j] = tempY;
        }
    }

    private static double calculateBatchLoss(double[][] predictions, double[][] labels, Loss lossFunction) {
        double totalLoss = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            totalLoss += lossFunction.calculate_loss(labels[i], predictions[i]);
        }
        return totalLoss / predictions.length;
    }

    private static int calculateCorrectPredictions(double[][] predictions, double[][] labels) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            int predictedClass = argmax(predictions[i]);
            int actualClass = argmax(labels[i]);
            if (predictedClass == actualClass) {
                correct++;
            }
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

    private static void addRegularizationGradients(Backward backward, double[][] W1, double[][] W2, double[][] W3, double lambda) {
        // Add L2 regularization gradients
        double[][] regGrad1 = Techniques.weightDecayGradient(W1, lambda);
        double[][] regGrad2 = Techniques.weightDecayGradient(W2, lambda);
        double[][] regGrad3 = Techniques.weightDecayGradient(W3, lambda);

        // Add to existing gradients
        backward.dW1 = Matrix_Operations.add(backward.dW1, regGrad1);
        backward.dW2 = Matrix_Operations.add(backward.dW2, regGrad2);
        backward.dW3 = Matrix_Operations.add(backward.dW3, regGrad3);
    }

    private static double evaluateModel(Forward forward, double[][] X_test, double[][] Y_test,
                                        double[][] W1, double[][] b1, double[][] W2, double[][] b2,
                                        double[][] W3, double[][] b3) {
        Techniques.clearCache();
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
            double[][] sampleInput = {X_test[sampleIdx]};

            Techniques.clearCache();
            double[][] prediction = forward.forward(sampleInput, W1, b1, W2, b2, W3, b3);

            int predictedClass = argmax(prediction[0]);
            int actualClass = actualLabels[sampleIdx];
            double confidence = prediction[0][predictedClass] * 100;

            String status = (predictedClass == actualClass) ? "✅" : "❌";
            System.out.printf("%s Sample %d: Predicted = %d, Actual = %d, Confidence = %.1f%%\n",
                    status, i + 1, predictedClass, actualClass, confidence);
        }
    }
}