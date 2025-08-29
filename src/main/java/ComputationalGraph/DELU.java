package ComputationalGraph;

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

    @Override
    public Tensor calculate(Tensor matrix) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) matrix.getData();
        for (Double oldValue : oldValues) {
            if (oldValue > this.xc) {
                values.add(oldValue);
            } else {
                values.add((Math.exp(this.a * oldValue) - 1) / this.b);
            }
        }
        return new Tensor(values, matrix.getShape());
    }

    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) value.getData();
        for (Double oldValue : oldValues) {
            if (oldValue > this.xc) {
                values.add(1.0);
            } else {
                values.add((oldValue * this.b + 1) * (this.a / this.b));
            }
        }
        return backward.hadamardProduct(new Tensor(values, value.getShape()));
    }
}
