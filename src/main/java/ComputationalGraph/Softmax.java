package ComputationalGraph;

import Math.Tensor;
import java.util.ArrayList;
import java.util.List;

public class Softmax implements Function {
    /**
     * Implements the Softmax activation function.
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
            List<Double> expValues = new ArrayList<>();
            double sum = 0.0;

            for (int j = 0; j < cols; j++) {
                double val = tensor.get(new int[]{i, j});
                double expVal = Math.exp(val);
                expValues.add(expVal);
                sum += expVal;
            }

            for (int j = 0; j < cols; j++) {
                result.set(new int[]{i, j}, expValues.get(j) / sum);
            }
        }

        return result;
    }

    @Override
    public Tensor derivative(Tensor tensor) {
        // Compute the softmax output first (like in the Python version)
        Tensor softmaxOutput = calculate(tensor);

        int[] shape = softmaxOutput.getShape();
        int rows = shape[0];
        int cols = shape[1];

        List<List<List<Double>>> initialData = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            List<List<Double>> rowList = new ArrayList<>();
            for (int j = 0; j < cols; j++) {
                List<Double> colList = new ArrayList<>();
                for (int k = 0; k < cols; k++) {
                    colList.add(0.0);
                }
                rowList.add(colList);
            }
            initialData.add(rowList);
        }

        Tensor result = new Tensor(initialData, new int[]{rows, cols, cols});

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double s_i = softmaxOutput.get(new int[]{i, j});
                for (int k = 0; k < cols; k++) {
                    double s_k = softmaxOutput.get(new int[]{i, k});
                    double value = (j == k) ? s_i * (1 - s_k) : -s_i * s_k;
                    result.set(new int[]{i, j, k}, value);
                }
            }
        }

        return result;
    }
}
