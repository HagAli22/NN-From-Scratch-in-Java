package org.example;

import java.io.*;
import java.util.concurrent.CompletableFuture;

/**
 * ModelIO class is responsible for:
 * - Saving model parameters (weights and biases) to a file
 * - Loading model parameters from a file
 * - Providing async versions of save and load
 * - Checking if a model file exists
 */
public class ModelIO {

    // Inner class to hold model parameters (weights and biases)
    public static class ModelParameters {
        public final double[][] W1, b1;
        public final double[][] W2, b2;
        public final double[][] W3, b3;

        public ModelParameters(double[][] W1, double[][] b1,
                               double[][] W2, double[][] b2,
                               double[][] W3, double[][] b3) {
            this.W1 = W1;
            this.b1 = b1;
            this.W2 = W2;
            this.b2 = b2;
            this.W3 = W3;
            this.b3 = b3;
        }
    }

    // Save model parameters into a file
    public static boolean saveModel(ModelParameters params, String filename) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(params.W1);
            oos.writeObject(params.b1);
            oos.writeObject(params.W2);
            oos.writeObject(params.b2);
            oos.writeObject(params.W3);
            oos.writeObject(params.b3);

            System.out.println("Model saved to " + filename);
            return true;
        } catch (IOException e) {
            System.err.println("Failed to save model: " + e.getMessage());
            return false;
        }
    }

    // Save model asynchronously (runs in background)
    public static CompletableFuture<Boolean> saveModelAsync(ModelParameters params, String filename) {
        return CompletableFuture.supplyAsync(() -> saveModel(params, filename));
    }

    // Load model parameters from a file
    public static ModelParameters loadModel(String filename) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            double[][] W1 = (double[][]) ois.readObject();
            double[][] b1 = (double[][]) ois.readObject();
            double[][] W2 = (double[][]) ois.readObject();
            double[][] b2 = (double[][]) ois.readObject();
            double[][] W3 = (double[][]) ois.readObject();
            double[][] b3 = (double[][]) ois.readObject();

            System.out.println("Model loaded from " + filename);
            return new ModelParameters(W1, b1, W2, b2, W3, b3);
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Failed to load model: " + e.getMessage());
            return null;
        }
    }

    // Load model asynchronously (runs in background)
    public static CompletableFuture<ModelParameters> loadModelAsync(String filename) {
        return CompletableFuture.supplyAsync(() -> loadModel(filename));
    }

    // Check if model file exists
    public static boolean modelExists(String filename) {
        File file = new File(filename);
        return file.exists() && file.isFile();
    }
}
