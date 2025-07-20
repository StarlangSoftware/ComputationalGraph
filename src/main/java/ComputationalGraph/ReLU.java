package ComputationalGraph;

import Math.Tensor;
import java.util.ArrayList;
import java.util.List;

public class ReLU implements Function {

    /**
     * Computes the ReLU activation for the given tensor.
     */
    @Override
    public Tensor calculate(Tensor tensor) {
        int[] shape = tensor.getShape();
        List<Double> resultData = new ArrayList<>();
        int totalElements = tensor.getData().size();
        
        for (int i = 0; i < totalElements; i++) {
            int[] indices = tensor.unflattenIndex(i, tensor.computeStrides(shape));
            double val = tensor.getValue(indices);
            resultData.add(Math.max(0, val));
        }
        
        return new Tensor(resultData, shape);
    }

    /**
     * Computes the derivative of the ReLU function.
     * Assumes input is the raw pre-activation tensor.
     */
    @Override
    public Tensor derivative(Tensor tensor) {
        int[] shape = tensor.getShape();
        List<Double> resultData = new ArrayList<>();
        int totalElements = tensor.getData().size();
        
        for (int i = 0; i < totalElements; i++) {
            int[] indices = tensor.unflattenIndex(i, tensor.computeStrides(shape));
            double val = tensor.getValue(indices);
            resultData.add(val > 0 ? 1.0 : 0.0);
        }
        
        return new Tensor(resultData, shape);
    }
}
