package ComputationalGraph;

import java.io.Serializable;
import java.util.ArrayList;

import Math.*;

public class Subtract implements Function, Serializable {
    @Override
    public Tensor calculate(Tensor matrix) {
        int[] shape = matrix.getShape();
        int rows = shape[0];
        int cols = shape[1];
        ArrayList<ArrayList<Double>> initialData = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            ArrayList<Double> row = new ArrayList<>();
            for (int j = 0; j < cols; j++) {
                row.add(0.0);
            }
            initialData.add(row);
        }
        Tensor result = new Tensor(initialData, shape);
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                result.set(new int[]{i, j}, -matrix.getValue(new int[]{i, j}));
            }
        }
        return result;
    }

    @Override
    public Tensor derivative(Tensor matrix) {
        return calculate(matrix);
    }
}
