package org.example;

import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

/**
 * Controller class for the main UI.
 * Handles image upload, prediction, and displaying results.
 */
public class MainController implements Initializable {

    @FXML private Button predictButton;
    @FXML private Button uploadImageButton;
    @FXML private TextArea resultArea;
    @FXML private Label statusLabel;
    @FXML private ProgressBar progressBar;
    @FXML private VBox inputContainer;
    @FXML private ImageView uploadedImageView;

    private final NeuralNetworkService neuralNetworkService;
    private Stage primaryStage;
    private File selectedImageFile;

    public MainController() {
        this.neuralNetworkService = new NeuralNetworkService();
    }

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        setupInitialState();
        setupEventHandlers();
    }

    /**
     * Setup initial UI state.
     */
    private void setupInitialState() {
        predictButton.setDisable(true);
        progressBar.setVisible(false);
        statusLabel.setText("Ready");
        resultArea.setEditable(false);
    }

    /**
     * Bind event handlers for buttons.
     */
    private void setupEventHandlers() {
        predictButton.setOnAction(e -> makePrediction());
        uploadImageButton.setOnAction(e -> chooseImage());
    }

    /**
     * Opens file chooser to select an image.
     */
    private void chooseImage() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Select Image");
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.jpeg")
        );

        File file = fileChooser.showOpenDialog(primaryStage);
        if (file != null) {
            selectedImageFile = file;
            try {
                Image img = new Image(new FileInputStream(file));
                uploadedImageView.setImage(img);
                predictButton.setDisable(false);
            } catch (IOException ex) {
                showAlert(Alert.AlertType.ERROR, "Error", "Cannot load image.");
            }
        }
    }

    /**
     * Makes prediction using the uploaded image.
     */
    @FXML
    private void makePrediction() {
        if (selectedImageFile == null) {
            showAlert(Alert.AlertType.WARNING, "Warning", "Please upload an image first.");
            return;
        }

        showLoadingState("Making prediction...");

        Task<NeuralNetworkService.PredictionResult> task = new Task<>() {
            @Override
            protected NeuralNetworkService.PredictionResult call() throws Exception {
                double[] input = neuralNetworkService.imageToMNISTInput(selectedImageFile);
                return neuralNetworkService.predictWithConfidence(input);
            }
        };

        task.setOnSucceeded(e -> {
            hideLoadingState();
            displayPredictionResult(task.getValue());
        });

        task.setOnFailed(e -> {
            hideLoadingState();
            showAlert(Alert.AlertType.ERROR, "Prediction Error", "Failed to predict the digit.");
        });

        new Thread(task).start();
    }

    /**
     * Display the prediction result in the text area.
     */
    private void displayPredictionResult(NeuralNetworkService.PredictionResult result) {
        resultArea.setText("Predicted Digit: " + result.predictedClass);
    }

    /**
     * Show loading state while prediction is running.
     */
    private void showLoadingState(String message) {
        statusLabel.setText(message);
        progressBar.setVisible(true);
        predictButton.setDisable(true);
    }

    /**
     * Hide loading state and restore buttons.
     */
    private void hideLoadingState() {
        progressBar.setVisible(false);
        predictButton.setDisable(selectedImageFile == null);
    }

    /**
     * Show alert dialog with message.
     */
    private void showAlert(Alert.AlertType type, String title, String message) {
        Alert alert = new Alert(type);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }

    public void setPrimaryStage(Stage primaryStage) {
        this.primaryStage = primaryStage;
    }
}
