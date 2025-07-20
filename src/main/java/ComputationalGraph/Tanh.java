package ComputationalGraph;

import Math.Tensor;
import java.util.ArrayList;
import java.util.List;

public class Tanh implements Function {

    /**
     * Computes the Tanh activation for the given tensor.
     */
    @Override
    public Tensor calculate(Tensor tensor) {
        int[] shape = tensor.getShape();
        List<Double> resultData = new ArrayList<>();
        int totalElements = tensor.getData().size();
        
        for (int i = 0; i < totalElements; i++) {
            int[] indices = tensor.unflattenIndex(i, tensor.computeStrides(shape));
            double val = tensor.getValue(indices);
            resultData.add(Math.tanh(val));
        }
        
        return new Tensor(resultData, shape);
    }

    /**
     * Computes the derivative of the Tanh function.
     * Assumes input is tanh(x), not raw x.
     */
    @Override
    public Tensor derivative(Tensor tensor) {
        int[] shape = tensor.getShape();
        List<Double> resultData = new ArrayList<>();
        int totalElements = tensor.getData().size();
        
        for (int i = 0; i < totalElements; i++) {
            int[] indices = tensor.unflattenIndex(i, tensor.computeStrides(shape));
            double tanhVal = tensor.getValue(indices);
            resultData.add(1 - tanhVal * tanhVal);
        }
        
        return new Tensor(resultData, shape);
    }
}
