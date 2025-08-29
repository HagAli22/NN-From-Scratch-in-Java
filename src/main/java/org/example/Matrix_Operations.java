package org.example;

public class Matrix_Operations {

    // ---------------- Matrix Operations ---------------- //

    // Addition
    public static double[][] add(double[][] a, double[][] b) {
        validateSameDimension(a, b);
        int rows = a.length, cols = a[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i][j] = a[i][j] + b[i][j];

        return result;
    }

    // Subtraction
    public static double[][] subtract(double[][] a, double[][] b) {
        validateSameDimension(a, b);
        int rows = a.length, cols = a[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i][j] = a[i][j] - b[i][j];

        return result;
    }

    // Scalar Multiplication
    public static double[][] scalarMultiply(double[][] a, double scalar) {
        int rows = a.length, cols = a[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i][j] = a[i][j] * scalar;

        return result;
    }

    // Matrix Multiplication
    public static double[][] multiply(double[][] a, double[][] b) {
        if (a[0].length != b.length)
            throw new IllegalArgumentException("Invalid matrix dimensions for multiplication.");

        int rows = a.length, cols = b[0].length, common = a[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                for (int k = 0; k < common; k++)
                    result[i][j] += a[i][k] * b[k][j];

        return result;
    }

    // Dot Product (for vectors)
    public static double dot(double[] a, double[] b) {
        if (a.length != b.length)
            throw new IllegalArgumentException("Vectors must be of same length.");

        double sum = 0;
        for (int i = 0; i < a.length; i++)
            sum += a[i] * b[i];

        return sum;
    }

    // Transpose
    public static double[][] transpose(double[][] a) {
        int rows = a.length, cols = a[0].length;
        double[][] result = new double[cols][rows];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[j][i] = a[i][j];

        return result;
    }

    // Determinant
    public static double determinant(double[][] a) {
        if (a.length != a[0].length)
            throw new IllegalArgumentException("Matrix must be square.");

        int n = a.length;

        if (n == 1) return a[0][0];
        if (n == 2) return a[0][0] * a[1][1] - a[0][1] * a[1][0];

        double det = 0;
        for (int col = 0; col < n; col++) {
            det += Math.pow(-1, col) * a[0][col] * determinant(minor(a, 0, col));
        }
        return det;
    }

    // Helper to compute minor matrix
    private static double[][] minor(double[][] a, int row, int col) {
        int n = a.length;
        double[][] result = new double[n - 1][n - 1];

        int r = 0;
        for (int i = 0; i < n; i++) {
            if (i == row) continue;
            int c = 0;
            for (int j = 0; j < n; j++) {
                if (j == col) continue;
                result[r][c] = a[i][j];
                c++;
            }
            r++;
        }
        return result;
    }

    // Inverse using adjoint and determinant
    public static double[][] inverse(double[][] a) {
        int n = a.length;
        if (n != a[0].length)
            throw new IllegalArgumentException("Matrix must be square.");

        double det = determinant(a);
        if (det == 0)
            throw new ArithmeticException("Matrix is singular and cannot be inverted.");

        double[][] adj = adjoint(a);
        double[][] inv = new double[n][n];

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                inv[i][j] = adj[i][j] / det;

        return inv;
    }

    // Adjoint matrix
    private static double[][] adjoint(double[][] a) {
        int n = a.length;
        double[][] adj = new double[n][n];

        if (n == 1) {
            adj[0][0] = 1;
            return adj;
        }

        int sign;
        double[][] temp;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                temp = minor(a, i, j);
                sign = ((i + j) % 2 == 0) ? 1 : -1;
                adj[j][i] = sign * determinant(temp);
            }
        }
        return adj;
    }
    // Helper: Find max value in a row
    static double findMax(double[] row) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : row) {
            if (v > max) max = v;
        }
        return max;
}
    // Helper: Compute exp(x - max) for each element and return their sum
    static double sumExp(double[] row, double max, double[] expRow) {
        double sum = 0;
        for (int j = 0; j < row.length; j++) {
            expRow[j] = Math.exp(row[j] - max);
            sum += expRow[j];
        }
        return sum;
}

    // Flatten
    public static double[] flatten(double[][] a) {
        int rows = a.length, cols = a[0].length;
        double[] flat = new double[rows * cols];
        int index = 0;

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                flat[index++] = a[i][j];

        return flat;
    }

    // Reshape
    public static double[][] reshape(double[] flat, int rows, int cols) {
        if (flat.length != rows * cols)
            throw new IllegalArgumentException("Invalid reshape dimensions.");

        double[][] reshaped = new double[rows][cols];
        int index = 0;

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                reshaped[i][j] = flat[index++];

        return reshaped;
    }

    // ---------------- Validation ---------------- //

    private static void validateSameDimension(double[][] a, double[][] b) {
        if (a.length != b.length || a[0].length != b[0].length)
            throw new IllegalArgumentException("Matrices must have the same dimensions.");
    }
}

