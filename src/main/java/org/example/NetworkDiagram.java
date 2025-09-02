package org.example;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.scene.effect.DropShadow;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Line;
import javafx.stage.Stage;

import java.util.ArrayList;
import java.util.List;

public class NetworkDiagram extends Application {

    private static final int SAMPLE_INPUT_SIZE = 15;  // Sample size for Input Layer
    private static final int SAMPLE_HIDDEN1_SIZE = 10; // Sample size for Hidden Layer 1
    private static final int SAMPLE_HIDDEN2_SIZE = 7; // Sample size for Hidden Layer 2
    private static final int SAMPLE_OUTPUT_SIZE = 5;  // Sample size for Output Layer

    @Override
    public void start(Stage primaryStage) {
        Pane root = new Pane();
        Scene scene = new Scene(root, 800, 400); // Suitable size for sample
        drawNetworkDiagram(root);
        primaryStage.setTitle("Sample Neural Network Architecture");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public void drawNetworkDiagram(Pane pane) {
        pane.getChildren().clear();
        double paneWidth = pane.getWidth() > 0 ? pane.getWidth() : 800;
        double paneHeight = pane.getHeight() > 0 ? pane.getHeight() : 400;
        double layerSpacing = paneWidth / 6; // 5 spaces between 6 layers (including margins)
        double nodeRadius = 15; // Increased radius for better visibility
        double verticalSpacing = paneHeight / (Math.max(SAMPLE_OUTPUT_SIZE, Math.max(SAMPLE_HIDDEN1_SIZE, SAMPLE_HIDDEN2_SIZE)) + 2);

        // Lists to store nodes for connections
        List<Circle> inputNodes = new ArrayList<>();
        List<Circle> hidden1Nodes = new ArrayList<>();
        List<Circle> hidden2Nodes = new ArrayList<>();
        List<Circle> outputNodes = new ArrayList<>();

        // Input Layer (sample of 5 nodes)
        double xInput = layerSpacing;
        for (int i = 0; i < SAMPLE_INPUT_SIZE; i++) {
            double y = i * verticalSpacing + verticalSpacing;
            Circle node = new Circle(xInput, y, nodeRadius);
            node.setFill(Color.web("#008B8B")); // Dark Cyan
            node.setStroke(Color.BLACK);
            node.setEffect(new DropShadow(5, Color.BLACK)); // Add shadow
            pane.getChildren().add(node);
            inputNodes.add(node);
        }
        pane.getChildren().add(new Label("Input- 28x28 -forMNIST") {
            { setLayoutX(xInput - 70); setLayoutY(10); setStyle("-fx-font-size: 12; -fx-font-weight: bold; -fx-text-fill: #008B8B;"); }
        });

        // Hidden Layer 1 (sample of 5 nodes)
        double xHidden1 = 2 * layerSpacing;
        for (int i = 0; i < SAMPLE_HIDDEN1_SIZE; i++) {
            double y = i * verticalSpacing + verticalSpacing;
            Circle node = new Circle(xHidden1, y, nodeRadius);
            node.setFill(Color.web("#20B2AA")); // Teal
            node.setStroke(Color.BLACK);
            node.setEffect(new DropShadow(5, Color.BLACK));
            pane.getChildren().add(node);
            hidden1Nodes.add(node);

            // Connect to all Input Layer nodes
            for (Circle inputNode : inputNodes) {
                Line line = new Line(inputNode.getCenterX(), inputNode.getCenterY(),
                        node.getCenterX(), node.getCenterY());
                line.setStroke(Color.DARKGRAY);
                line.setStrokeWidth(0.5);
                pane.getChildren().add(line);
            }
        }
        pane.getChildren().add(new Label("Hidden-Layer1 (128)") {
            { setLayoutX(xHidden1 - 80); setLayoutY(10); setStyle("-fx-font-size: 12; -fx-font-weight: bold; -fx-text-fill: #20B2AA;"); }
        });

        // Hidden Layer 2 (sample of 5 nodes)
        double xHidden2 = 3 * layerSpacing;
        for (int i = 0; i < SAMPLE_HIDDEN2_SIZE; i++) {
            double y = i * verticalSpacing + verticalSpacing;
            Circle node = new Circle(xHidden2, y, nodeRadius);
            node.setFill(Color.web("#DAA520")); // Goldenrod
            node.setStroke(Color.BLACK);
            node.setEffect(new DropShadow(5, Color.BLACK));
            pane.getChildren().add(node);
            hidden2Nodes.add(node);

            // Connect to all Hidden Layer 1 nodes
            for (Circle hidden1Node : hidden1Nodes) {
                Line line = new Line(hidden1Node.getCenterX(), hidden1Node.getCenterY(),
                        node.getCenterX(), node.getCenterY());
                line.setStroke(Color.DARKGRAY);
                line.setStrokeWidth(0.5);
                pane.getChildren().add(line);
            }
        }
        pane.getChildren().add(new Label("Hidden-Layer2 (64)") {
            { setLayoutX(xHidden2 - 80); setLayoutY(10); setStyle("-fx-font-size: 12; -fx-font-weight: bold; -fx-text-fill: #DAA520;"); }
        });

        // Output Layer (sample of 5 nodes)
        double xOutput = 4 * layerSpacing;
        for (int i = 0; i < SAMPLE_OUTPUT_SIZE; i++) {
            double y = i * verticalSpacing + verticalSpacing;
            Circle node = new Circle(xOutput, y, nodeRadius);
            node.setFill(Color.web("#DC143C")); // Crimson
            node.setStroke(Color.BLACK);
            node.setEffect(new DropShadow(5, Color.BLACK));
            pane.getChildren().add(node);
            outputNodes.add(node);

            // Connect to all Hidden Layer 2 nodes
            for (Circle hidden2Node : hidden2Nodes) {
                Line line = new Line(hidden2Node.getCenterX(), hidden2Node.getCenterY(),
                        node.getCenterX(), node.getCenterY());
                line.setStroke(Color.DARKGRAY);
                line.setStrokeWidth(0.5);
                pane.getChildren().add(line);
            }
        }
        pane.getChildren().add(new Label("Output-Layer (10) 0 to 9") {
            { setLayoutX(xOutput - 70); setLayoutY(10); setStyle("-fx-font-size: 12; -fx-font-weight: bold; -fx-text-fill: #DC143C;"); }
        });

        // Update drawing when pane size changes
        pane.widthProperty().addListener((obs, oldVal, newVal) -> {
            pane.getChildren().clear();
            drawNetworkDiagram(pane);
        });
        pane.heightProperty().addListener((obs, oldVal, newVal) -> {
            pane.getChildren().clear();
            drawNetworkDiagram(pane);
        });
    }

    public static void main(String[] args) {
        launch(args); // Launch the JavaFX application
    }
}