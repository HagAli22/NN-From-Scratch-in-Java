package org.example;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 * NeuralNetworkService is responsible for:
 * - Loading a trained model
 * - Running predictions
 * - Converting images into MNIST-like input
 */
public class NeuralNetworkService {
    private ModelIO.ModelParameters currentModel;
    private boolean isModelLoaded = false;
    private Forward forwardPass;
    private static final String DEFAULT_MODEL_PATH = "saved_model.dat";

    public NeuralNetworkService() {
        this.forwardPass = new Forward();
        loadDefaultModel();
    }

    // Load the default model from file
    private void loadDefaultModel() {
        currentModel = ModelIO.loadModel(DEFAULT_MODEL_PATH);
        isModelLoaded = (currentModel != null);
        if (!isModelLoaded) {
            System.err.println("⚠️ Warning: Failed to load default model from " + DEFAULT_MODEL_PATH);
        }
    }

    // Predict using the loaded model
    public double[] predict(double[] input) {
        if (!isModelLoaded || currentModel == null) {
            throw new IllegalStateException("Model not loaded.");
        }

        double[][] inputBatch = new double[1][input.length];
        inputBatch[0] = input;

        double[][] output = forwardPass.forward(
                inputBatch,
                currentModel.W1, currentModel.b1,
                currentModel.W2, currentModel.b2,
                currentModel.W3, currentModel.b3
        );

        return output[0];
    }

    // Predict with confidence and return probabilities
    public PredictionResult predictWithConfidence(double[] input) {
        double[] probabilities = predict(input);

        int predictedClass = 0;
        double maxProb = probabilities[0];

        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                predictedClass = i;
            }
        }

        return new PredictionResult(predictedClass, maxProb, probabilities);
    }

    // Convert an image file into a normalized MNIST-like input vector
    public double[] imageToMNISTInput(File imageFile) throws Exception {
        BufferedImage original = ImageIO.read(imageFile);

        // 1. Convert to grayscale
        BufferedImage gray = new BufferedImage(original.getWidth(), original.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = gray.createGraphics();
        g.drawImage(original, 0, 0, null);
        g.dispose();

        // 2. Detect background brightness
        long sum = 0;
        for (int y = 0; y < gray.getHeight(); y++) {
            for (int x = 0; x < gray.getWidth(); x++) {
                sum += gray.getRGB(x, y) & 0xFF;
            }
        }
        double avg = sum / (double) (gray.getWidth() * gray.getHeight());
        boolean invertColors = avg > 127; // invert if background is bright

        // 3. Find bounding box of the digit
        int minX = gray.getWidth(), minY = gray.getHeight();
        int maxX = 0, maxY = 0;
        for (int y = 0; y < gray.getHeight(); y++) {
            for (int x = 0; x < gray.getWidth(); x++) {
                int pixel = gray.getRGB(x, y) & 0xFF;
                if (invertColors) pixel = 255 - pixel;
                if (pixel > 30) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        int width = maxX - minX + 1;
        int height = maxY - minY + 1;
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("No digit found in the image.");
        }

        BufferedImage cropped = gray.getSubimage(minX, minY, width, height);

        // 4. Resize to 20x20 while keeping aspect ratio
        int targetSize = 20;
        BufferedImage resized = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2 = resized.createGraphics();
        g2.setColor(Color.BLACK);
        g2.fillRect(0, 0, targetSize, targetSize);

        double aspect = width / (double) height;
        int newWidth = aspect > 1 ? targetSize : (int) (targetSize * aspect);
        int newHeight = aspect > 1 ? (int) (targetSize / aspect) : targetSize;

        int xOffset = (targetSize - newWidth) / 2;
        int yOffset = (targetSize - newHeight) / 2;

        g2.drawImage(cropped, xOffset, yOffset, newWidth, newHeight, null);
        g2.dispose();

        // 5. Place inside 28x28 canvas
        BufferedImage finalImg = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g3 = finalImg.createGraphics();
        g3.setColor(Color.BLACK);
        g3.fillRect(0, 0, 28, 28);
        g3.drawImage(resized, 4, 4, null);
        g3.dispose();

        // 6. Flatten and normalize pixels
        double[] input = new double[28 * 28];
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int val = finalImg.getRGB(x, y) & 0xFF;
                if (invertColors) val = 255 - val;
                input[y * 28 + x] = val / 255.0;
            }
        }

        return input;
    }

    public boolean isModelLoaded() {
        return isModelLoaded;
    }

    // Class to hold prediction result
    public static class PredictionResult {
        public final int predictedClass;
        public final double confidence;
        public final double[] allProbabilities;

        public PredictionResult(int predictedClass, double confidence, double[] allProbabilities) {
            this.predictedClass = predictedClass;
            this.confidence = confidence;
            this.allProbabilities = allProbabilities;
        }
    }
}
