package org.example;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.stage.Stage;

/**
 * Entry point for the Neural Network JavaFX application.
 * Loads the FXML UI, sets up the stage, and starts the app.
 */
public class NeuralNetworkApp extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {
        // Load FXML layout
        FXMLLoader loader = new FXMLLoader(getClass().getResource("/org/example/MainView.fxml"));
        Parent root = loader.load();

        // Get controller and inject the stage
        MainController controller = loader.getController();
        controller.setPrimaryStage(primaryStage);

        // Create scene
        Scene scene = new Scene(root, 800, 600);

        // Optional: add custom CSS
        // scene.getStylesheets().add(getClass().getResource("/css/styles.css").toExternalForm());

        // Configure primary stage
        primaryStage.setTitle("Neural Network Desktop App");
        primaryStage.setScene(scene);
        primaryStage.setMinWidth(600);
        primaryStage.setMinHeight(500);

        // Optional: set app icon
        try {
            primaryStage.getIcons().add(new Image(getClass().getResourceAsStream("/images/icon.png")));
        } catch (Exception e) {
            System.out.println("Icon not found, using default");
        }

        // Show the stage
        primaryStage.show();

        // Shutdown hook for cleanup
        primaryStage.setOnCloseRequest(e -> {
            System.out.println("Application closing...");
            System.exit(0);
        });
    }

    public static void main(String[] args) {
        // Enable JavaFX preloader for better startup performance
        System.setProperty("javafx.preloader", "true");

        // Launch the JavaFX application
        launch(args);
    }
}
