package ComputationalGraph;

import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class ReLU implements Function, Serializable {

    /**
     * Computes the ReLU activation for the given tensor.
     */
    @Override
    public Tensor calculate(Tensor tensor) {
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
                double val = tensor.getValue(new int[]{i, j});
                result.set(new int[]{i, j}, Math.max(0, val));
            }
        }
        return result;
    }

    /**
     * Computes the derivative of the ReLU function.
     * Assumes input is the raw pre-activation tensor.
     */
    @Override
    public Tensor derivative(Tensor tensor) {
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
                double val = tensor.getValue(new int[]{i, j});
                result.set(new int[]{i, j}, val > 0 ? 1.0 : 0.0);
            }
        }
        return result;
    }
}
