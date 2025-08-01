package ComputationalGraph;

import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class Softmax implements Function, Serializable {
    /**
     * Implements the Softmax activation function.
     */

    @Override
    public Tensor calculate(Tensor tensor) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) tensor.getData();
        int lastDimensionSize = tensor.getShape()[tensor.getShape().length - 1];
        double sum = 0.0;
        ArrayList<Double> sumList = new ArrayList<>();
        for (int i = 0; i < oldValues.size(); i++) {
            sum += Math.exp(oldValues.get(i));
            if ((i + 1) % lastDimensionSize == 0) {
                sumList.add(sum);
                sum = 0.0;
            }
        }
        for (int i = 0; i < oldValues.size(); i++) {
            values.add(Math.exp(oldValues.get(i)) / sumList.get(i / lastDimensionSize));
        }
        return new Tensor(values, tensor.getShape());
    }

    @Override
    public Tensor derivative(Tensor tensor) {
        int[] shape = new int[tensor.getShape().length];
        if (tensor.getShape().length - 1 >= 0) {
            System.arraycopy(tensor.getShape(), 1, shape, 0, tensor.getShape().length - 1);
        }
        int lastDimensionSize = tensor.getShape()[tensor.getShape().length - 1];
        shape[shape.length - 1] = lastDimensionSize;
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) tensor.getData();
        for (int i = 0; i < oldValues.size(); i++) {
            double v1 = oldValues.get(i);
            int startIndex = i / lastDimensionSize;
            for (int j = 0; j < lastDimensionSize; j++) {
                double v2 = oldValues.get(startIndex + j);
                if (v1 % lastDimensionSize == j) {
                    values.add(v1 * (1 - v2));
                } else {
                    values.add(-v1 * v2);
                }
            }
        }
        return new Tensor(values, shape);
    }
}
