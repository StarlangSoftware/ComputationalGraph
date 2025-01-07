package ComputationalGraph;

import Math.*;

public class Softmax implements Function {

    @Override
    public Matrix calculate(Matrix matrix) {
        Matrix result = new Matrix(matrix.getRow(), matrix.getColumn());
        for (int i = 0; i < matrix.getRow(); i++) {
            double sum = 0.0;
            for (int k = 0; k < matrix.getColumn(); k++) {
                sum += Math.exp(matrix.getValue(i, k));
            }
            for (int k = 0; k < matrix.getColumn(); k++) {
                result.setValue(i, k, Math.exp(matrix.getValue(i, k)) / sum);
            }
        }
        return result;
    }

    @Override
    public Matrix derivative(Matrix matrix) {
        return null;
    }
}
