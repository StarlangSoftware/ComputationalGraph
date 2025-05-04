package ComputationalGraph;

import Math.Tensor;
import java.util.ArrayList;
import java.util.List;

public class Tanh implements Function {
    /**
     * Implements the Tanh activation function.
     */

    @Override
    public Tensor calculate(Tensor tensor) {
        /**
         * Computes the Tanh activation for the given tensor.
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
                result.set(new int[]{i, j}, Math.tanh(val));
            }
        }

        return result;
    }

    @Override
    public Tensor derivative(Tensor tensor) {
        /**
         * Computes the derivative of the Tanh function.
         * Assumes input is tanh(x), not raw x.
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
                double tanhVal = tensor.get(new int[]{i, j});
                result.set(new int[]{i, j}, 1 - tanhVal * tanhVal);
            }
        }

        return result;
    }
}
