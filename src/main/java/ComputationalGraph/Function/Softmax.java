package ComputationalGraph.Function;

import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class Softmax implements Function, Serializable {

    /**
     * Computes the Softmax activation for the given tensor.
     * @param tensor The tensor whose values are to be computed.
     * @return Softmax(x).
     */
    @Override
    public FunctionResults calculate(Tensor tensor) {
        double[] oldValues = tensor.getData();
        double[] values = new double[oldValues.length];
        int lastDimensionSize = tensor.getShape()[tensor.getShape().length - 1];
        double sum = 0.0;
        ArrayList<Double> sumList = new ArrayList<>();
        for (int i = 0; i < oldValues.length; i++) {
            sum += Math.exp(oldValues[i]);
            if ((i + 1) % lastDimensionSize == 0) {
                sumList.add(sum);
                sum = 0.0;
            }
        }
        for (int i = 0; i < oldValues.length; i++) {
            values[i] = Math.exp(oldValues[i]) / sumList.get(i / lastDimensionSize);
        }
        return new FunctionResults(new Tensor(values, tensor.getShape()));
    }

    /**
     * Computes the derivative of the Softmax activation function.
     * @param tensor output of the Softmax(x).
     * @param backward Backward tensor.
     * @return Gradient value of the corresponding node.
     */
    @Override
    public Tensor derivative(Tensor tensor, Tensor backward) {
        int lastDimensionSize = tensor.getShape()[tensor.getShape().length - 1];
        ArrayList<Double> values = new ArrayList<>();
        double[] oldValuesTensor = tensor.getData();
        double[] oldValuesBackward = backward.getData();
        double total = 0.0;
        for (int i = 0; i < oldValuesTensor.length; i++) {
            total += oldValuesTensor[i] * oldValuesBackward[i];
            if ((i + 1) % lastDimensionSize == 0) {
                int startIndex = i / lastDimensionSize;
                for (int j = 0; j < lastDimensionSize; j++) {
                    values.add(oldValuesBackward[startIndex * lastDimensionSize + j] - total);
                }
                total = 0.0;
            }
        }
        return tensor.hadamardProduct(new Tensor(values, tensor.getShape()));
    }
}
