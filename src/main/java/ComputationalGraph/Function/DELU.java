package ComputationalGraph.Function;

import java.io.Serializable;
import java.util.ArrayList;

import Math.Tensor;

public class DELU implements Function, Serializable {

    private final double a;
    private final double b;
    private final double xc;

    public DELU(double a, double b, double xc) {
        this.a = a;
        this.b = b;
        this.xc = xc;
    }

    public DELU() {
        this.a = 1.0;
        this.b = 2.0;
        this.xc = 1.25643;
    }

    /**
     * Computes the DELU activation for the given value tensor.
     * @param value The tensor whose values are to be computed.
     * @return DELU(x).
     */
    @Override
    public Tensor calculate(Tensor value) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) value.getData();
        for (Double oldValue : oldValues) {
            if (oldValue > this.xc) {
                values.add(oldValue);
            } else {
                values.add((Math.exp(this.a * oldValue) - 1) / this.b);
            }
        }
        return new Tensor(values, value.getShape());
    }

    /**
     * Computes the derivative of the DELU activation function.
     * @param value output of the DELU(x).
     * @param backward Backward tensor.
     * @return Gradient value of the corresponding node.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) value.getData();
        ArrayList<Double> backwardValues = (ArrayList<Double>) backward.getData();
        for (int i = 0; i < oldValues.size(); i++) {
            Double backwardValue = backwardValues.get(i);
            Double oldValue = oldValues.get(i);
            if (oldValue > this.xc) {
                values.add(backwardValue);
            } else {
                values.add(backwardValue * ((oldValue * this.b + 1) * (this.a / this.b)));
            }
        }
        return new Tensor(values, value.getShape());
    }
}
