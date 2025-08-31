package org.example;

/**
 * A utility class for performing various matrix operations.
 */
public class Matrix_Operations {

    // ---------------- Matrix Operations ---------------- //

    /**
     * Adds two matrices of the same dimensions.
     * @param a First matrix
     * @param b Second matrix
     * @return Resulting matrix from element-wise addition
     * @throws IllegalArgumentException if dimensions do not match
     */
    public static double[][] add(double[][] a, double[][] b) {
        validateSameDimension(a, b);
        double[][] result = createMatrix(a.length, a[0].length);

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }

    /**
     * Broadcasts a bias vector to match batch size.
     * @param bias Bias matrix [1 x features]
     * @param batchSize Number of samples in batch
     * @return Broadcasted bias [batchSize x features]
     */
    public static double[][] broadcastBias(double[][] bias, int batchSize) {
        int features = bias[0].length;
        double[][] broadcasted = createMatrix(batchSize, features);

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < features; j++) {
                broadcasted[i][j] = bias[0][j];
            }
        }
        return broadcasted;
    }

    /**
     * Subtracts two matrices of the same dimensions.
     * @param a First matrix
     * @param b Second matrix
     * @return Resulting matrix from element-wise subtraction
     * @throws IllegalArgumentException if dimensions do not match
     */
    public static double[][] subtract(double[][] a, double[][] b) {
        validateSameDimension(a, b);
        double[][] result = createMatrix(a.length, a[0].length);

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }

    /**
     * Multiplies a matrix by a scalar.
     * @param a Input matrix
     * @param scalar Scalar value
     * @return Resulting matrix after scalar multiplication
     */
    public static double[][] scalarMultiply(double[][] a, double scalar) {
        double[][] result = createMatrix(a.length, a[0].length);

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                result[i][j] = a[i][j] * scalar;
            }
        }
        return result;
    }

    /**
     * Multiplies two matrices.
     * @param a First matrix
     * @param b Second matrix
     * @return Resulting matrix from matrix multiplication
     * @throws IllegalArgumentException if dimensions are incompatible
     */
    public static double[][] multiply(double[][] a, double[][] b) {
        if (a[0].length != b.length) {
            throw new IllegalArgumentException("Invalid matrix dimensions for multiplication.");
        }

        double[][] result = createMatrix(a.length, b[0].length);

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                for (int k = 0; k < a[0].length; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }

    /**
     * Computes the dot product of two vectors.
     * @param a First vector
     * @param b Second vector
     * @return Dot product value
     * @throws IllegalArgumentException if vectors have different lengths
     */
    public static double dot(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must be of same length.");
        }

        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /**
     * Transposes a matrix.
     * @param a Input matrix
     * @return Transposed matrix
     */
    public static double[][] transpose(double[][] a) {
        double[][] result = createMatrix(a[0].length, a.length);

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                result[j][i] = a[i][j];
            }
        }
        return result;
    }

    /**
     * Computes the determinant of a square matrix.
     * @param a Input square matrix
     * @return Determinant value
     * @throws IllegalArgumentException if matrix is not square
     */
    public static double determinant(double[][] a) {
        if (a.length != a[0].length) {
            throw new IllegalArgumentException("Matrix must be square.");
        }

        int n = a.length;

        if (n == 1) return a[0][0];
        if (n == 2) return a[0][0] * a[1][1] - a[0][1] * a[1][0];

        double det = 0;
        for (int col = 0; col < n; col++) {
            det += Math.pow(-1, col) * a[0][col] * determinant(minor(a, 0, col));
        }
        return det;
    }

    /**
     * Computes the inverse of a square matrix using adjoint and determinant.
     * @param a Input square matrix
     * @return Inverse matrix
     * @throws IllegalArgumentException if matrix is not square
     * @throws ArithmeticException if matrix is singular
     */
    public static double[][] inverse(double[][] a) {
        int n = a.length;
        if (n != a[0].length) {
            throw new IllegalArgumentException("Matrix must be square.");
        }

        double det = determinant(a);
        if (det == 0) {
            throw new ArithmeticException("Matrix is singular and cannot be inverted.");
        }

        double[][] adj = adjoint(a);
        double[][] inv = createMatrix(n, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inv[i][j] = adj[i][j] / det;
            }
        }
        return inv;
    }

    // ---------------- Helper Methods ---------------- //

    /**
     * Computes the minor matrix by excluding specified row and column.
     * @param a Input matrix
     * @param row Row to exclude
     * @param col Column to exclude
     * @return Minor matrix
     */
    private static double[][] minor(double[][] a, int row, int col) {
        int n = a.length;
        double[][] result = createMatrix(n - 1, n - 1);

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

    /**
     * Computes the adjoint (adjugate) matrix.
     * @param a Input square matrix
     * @return Adjoint matrix
     */
    private static double[][] adjoint(double[][] a) {
        int n = a.length;
        double[][] adj = createMatrix(n, n);

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

    /**
     * Finds the maximum value in a row for numerical stability.
     * @param row The input row array
     * @return The maximum value in the row
     */
    static double findMax(double[] row) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : row) {
            if (v > max) max = v;
        }
        return max;
    }

    /**
     * Computes exp(x - max) for each element and returns their sum for numerical stability.
     * @param row The input row array
     * @param max The maximum value in the row
     * @param expRow Array to store exponentiated values
     * @return Sum of exponentiated values
     */
    static double sumExp(double[] row, double max, double[] expRow) {
        double sum = 0;
        for (int j = 0; j < row.length; j++) {
            expRow[j] = Math.exp(row[j] - max);
            sum += expRow[j];
        }
        return sum;
    }

    /**
     * Flattens a matrix into a 1D array.
     * @param a Input matrix
     * @return Flattened array
     */
    public static double[] flatten(double[][] a) {
        int rows = a.length, cols = a[0].length;
        double[] flat = new double[rows * cols];
        int index = 0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flat[index++] = a[i][j];
            }
        }
        return flat;
    }

    /**
     * Reshapes a 1D array into a matrix.
     * @param flat Input flattened array
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Reshaped matrix
     * @throws IllegalArgumentException if dimensions are invalid
     */
    public static double[][] reshape(double[] flat, int rows, int cols) {
        if (flat.length != rows * cols) {
            throw new IllegalArgumentException("Invalid reshape dimensions.");
        }

        double[][] reshaped = createMatrix(rows, cols);
        int index = 0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                reshaped[i][j] = flat[index++];
            }
        }
        return reshaped;
    }

    // ---------------- Validation ---------------- //

    /**
     * Validates that two matrices have the same dimensions.
     * @param a First matrix
     * @param b Second matrix
     * @throws IllegalArgumentException if dimensions do not match
     */
    private static void validateSameDimension(double[][] a, double[][] b) {
        if (a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("Matrices must have the same dimensions.");
        }
    }

    // ---------------- Utility Methods ---------------- //

    /**
     * Creates a new matrix with specified dimensions.
     * @param rows Number of rows
     * @param cols Number of columns
     * @return New matrix
     */
    private static double[][] createMatrix(int rows, int cols) {
        return new double[rows][cols];
    }
}