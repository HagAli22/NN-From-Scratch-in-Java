package org.example;

/**
 * A utility class for mathematical computations.
 */
public class MathUtils {

    private static final double EPSILON = 1e-10;
    private static final int MAX_ITERATIONS = 100;

    /**
     * Computes the exponential function e^x using Taylor series.
     *v.1
     * @param x Input value
     * @return e^x
     * @throws IllegalArgumentException if x is NaN or Infinity
     */
    public static double exp(double x) {
        if (Double.isNaN(x) || Double.isInfinite(x)) {
            throw new IllegalArgumentException("exp input must not be NaN or Infinity: " + x);
        }
        double sum = 1.0, term = 1.0;
        for (int i = 1; i < MAX_ITERATIONS && Math.abs(term) > EPSILON; i++) {
            term *= x / i;
            sum += term;
        }
        return sum;
    }

    /**
     * Computes the natural logarithm using Newton-Raphson method.
     *
     * @param x Input value (must be positive)
     * @return ln(x)
     * @throws IllegalArgumentException if x <= 0, NaN, or Infinity
     */
    public static double log(double x) {
        if (x <= 0 || Double.isNaN(x) || Double.isInfinite(x)) {
            throw new IllegalArgumentException("Logarithm undefined for non-positive, NaN, or Infinity values: " + x);
        }
        double guess = x > 1 ? x - 1 : 0;
        for (int i = 0; i < MAX_ITERATIONS; i++) {
            double expGuess = exp(guess);
            double derivative = expGuess;
            double nextGuess = guess - (expGuess - x) / derivative;
            if (Math.abs(nextGuess - guess) < EPSILON) {
                return nextGuess;
            }
            guess = nextGuess;
        }
        return guess;
    }

    /**
     * Computes base raised to the power of exponent.
     *
     * @param base Base value
     * @param exponent Exponent value
     * @return base^exponent
     * @throws IllegalArgumentException if base <= 0 and exponent is non-integer, or if inputs are NaN/Infinity
     */
    public static double pow(double base, double exponent) {
        if (Double.isNaN(base) || Double.isInfinite(base) || Double.isNaN(exponent) || Double.isInfinite(exponent)) {
            throw new IllegalArgumentException("pow inputs must not be NaN or Infinity");
        }
        if (exponent == 0) return 1.0;
        if (base == 0 && exponent < 0) {
            throw new IllegalArgumentException("Undefined: zero raised to negative power");
        }
        if (base < 0 && exponent % 1 != 0) {
            throw new IllegalArgumentException("Negative base with non-integer exponent is undefined");
        }

        double result = 1.0;
        int intPart = (int) exponent;
        double fractionalPart = exponent - intPart;

        // Handle integer part
        double absBase = Math.abs(base);
        for (int i = 0; i < Math.abs(intPart); i++) {
            result *= absBase;
        }
        if (intPart < 0) {
            result = 1.0 / result;
        }

        // Handle fractional part using e^(fractional * ln(base))
        if (fractionalPart != 0) {
            result *= exp(fractionalPart * log(absBase));
        }

        return base < 0 && intPart % 2 != 0 ? -result : result;
    }
}
