package org.example;
import java.io.*;
import java.io.IOException;

class DataSet {
    private final int[][] images;
    private final int[] labels;

    public DataSet(int[][] images, int[] labels) {
        this.images = images;
        this.labels = labels;
    }

    public int[][] getImages() {
        return images;
    }

    public int[] getLabels() {
        return labels;
    }

    public int getSize() {
        return images.length;
    }
}

public class Load_dataset {

    private int[][] readImages(String file) throws IOException {
        DataInputStream dis =new DataInputStream(new FileInputStream(file));

        int magic = dis.readInt();
        int numImage = dis.readInt();
        int numRows = dis.readInt();
        int numCols = dis.readInt();

        int[][] images = new int[numImage][numRows*numCols];

        for (int i=0 ; i<numImage; i++){
            for (int j=0; j<numRows; j++){
                images[i][j] = dis.readUnsignedByte();
            }
        }
        dis.close();
        return images;
    }

    private int[] readLabels(String file) throws IOException{
        DataInputStream dis = new DataInputStream(new FileInputStream(file));
        int magic =dis.readInt();
        int numLabels=dis.readInt();

        int[] imageLabels =new int[numLabels];

        for (int i=0 ; i<numLabels; i++){
            imageLabels[i]=dis.readUnsignedByte();
        }
        dis.close();
        return imageLabels;
    }

    public DataSet loadData(String fileImages , String fileImagesLabels) throws IOException {

        int[][] images = readImages(fileImages);

        int[] imagesLabels = readLabels(fileImagesLabels);

        return new DataSet(images,imagesLabels);

    }




}