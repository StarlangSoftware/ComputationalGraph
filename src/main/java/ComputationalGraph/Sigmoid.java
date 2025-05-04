package ComputationalGraph;

import Math.Tensor;
import java.util.ArrayList;
import java.util.List;

public class Sigmoid implements Function {
    /**
     * Implements the Sigmoid activation function.
     */

    @Override
    public Tensor calculate(Tensor tensor) {
        /**
         * Computes the Sigmoid activation for the given tensor.
         */
        int[] shape = tensor.getShape();
        int rows = shape[0];
        int cols = shape[1];
        List<List<Double>> initialData = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < cols; j++) {
                row.add(0.0);
            }
            initialData.add(row);
        }
        Tensor result = new Tensor(initialData, shape);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double val = tensor.get(new int[]{i, j});
                double sigmoid = 1.0 / (1.0 + Math.exp(-val));
                result.set(new int[]{i, j}, sigmoid);
            }
        }

        return result;
    }

    @Override
    public Tensor derivative(Tensor tensor) {
        /**
         * Computes the derivative of the Sigmoid function.
         * Assumes `tensor` is the output of sigmoid(x).
         */
        int[] shape = tensor.getShape();
        int rows = shape[0];
        int cols = shape[1];
        List<List<Double>> initialData = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < cols; j++) {
                row.add(0.0);
            }
            initialData.add(row);
        }
        Tensor result = new Tensor(initialData, shape);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double sigmoidVal = tensor.get(new int[]{i, j});
                result.set(new int[]{i, j}, sigmoidVal * (1 - sigmoidVal));
            }
        }

        return result;
    }
}
