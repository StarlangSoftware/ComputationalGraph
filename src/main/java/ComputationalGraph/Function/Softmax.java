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
        ArrayList<Double> oldValuesTensor = (ArrayList<Double>) tensor.getData();
        ArrayList<Double> oldValuesBackward = (ArrayList<Double>) backward.getData();
        double total = 0.0;
        for (int i = 0; i < oldValuesTensor.size(); i++) {
            total += oldValuesTensor.get(i) * oldValuesBackward.get(i);
            if ((i + 1) % lastDimensionSize == 0) {
                int startIndex = i / lastDimensionSize;
                for (int j = 0; j < lastDimensionSize; j++) {
                    values.add(oldValuesBackward.get(startIndex * lastDimensionSize + j) - total);
                }
                total = 0.0;
            }
        }
        return tensor.hadamardProduct(new Tensor(values, tensor.getShape()));
    }
}
