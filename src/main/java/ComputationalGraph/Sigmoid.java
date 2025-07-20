package ComputationalGraph;

import Math.Tensor;
import java.util.ArrayList;
import java.util.List;

public class Sigmoid implements Function {

    /**
     * Computes the Sigmoid activation for the given tensor.
     */
    @Override
    public Tensor calculate(Tensor tensor) {
        int[] shape = tensor.getShape();
        List<Double> resultData = new ArrayList<>();
        int totalElements = tensor.getData().size();
        
        for (int i = 0; i < totalElements; i++) {
            int[] indices = tensor.unflattenIndex(i, tensor.computeStrides(shape));
            double val = tensor.getValue(indices);
            double sigmoid = 1.0 / (1.0 + Math.exp(-val));
            resultData.add(sigmoid);
        }
        
        return new Tensor(resultData, shape);
    }

    /**
     * Computes the derivative of the Sigmoid function.
     * Assumes `tensor` is the output of sigmoid(x).
     */
    @Override
    public Tensor derivative(Tensor tensor) {
        int[] shape = tensor.getShape();
        List<Double> resultData = new ArrayList<>();
        int totalElements = tensor.getData().size();
        
        for (int i = 0; i < totalElements; i++) {
            int[] indices = tensor.unflattenIndex(i, tensor.computeStrides(shape));
            double sigmoidVal = tensor.getValue(indices);
            resultData.add(sigmoidVal * (1 - sigmoidVal));
        }
        
        return new Tensor(resultData, shape);
    }
}
