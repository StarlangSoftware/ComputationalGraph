package ComputationalGraph.Function;

import java.io.Serializable;
import java.util.ArrayList;

import Math.Tensor;

public class Logarithm implements Function, Serializable {

    /**
     * Applies the natural logarithm function to each element of the input tensor.
     * @param value The tensor whose elements are to be transformed using the natural logarithm.
     * @return log(x) and x.
     */
    @Override
    public FunctionResults calculate(Tensor value) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) value.getData();
        for (Double oldValue : oldValues) {
            if (oldValue <= 0) {
                throw new IllegalArgumentException("Logarithm function input must be strictly positive. Found: " + oldValue);
            }
            values.add(Math.log(oldValue));
        }
        return new FunctionResults(new Tensor(values, value.getShape()), new Tensor(oldValues, value.getShape()));
    }

    /**
     * Computes the derivative of the Logarithm function.
     * @param value input of the Logarithm(x).
     * @param backward Backward tensor.
     * @return Gradient value of the corresponding node.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> tensorValues = (ArrayList<Double>) value.getData();
        ArrayList<Double> backwardValues = (ArrayList<Double>) backward.getData();
        for (int i = 0; i < tensorValues.size(); i++) {
            double val = tensorValues.get(i);
            double derivative = 1.0 / val;
            double backwardValue = backwardValues.get(i);
            values.add(derivative * backwardValue);
        }
        return new Tensor(values, value.getShape());
    }
}
