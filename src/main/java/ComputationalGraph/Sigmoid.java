package ComputationalGraph;

import Math.*;

public class Sigmoid implements Function {

    @Override
    public Matrix calculate(Matrix matrix) {
        Matrix result = new Matrix(matrix.getRow(), matrix.getColumn());
        for (int i = 0; i < matrix.getRow(); i++) {
            for (int j = 0; j < matrix.getColumn(); j++) {
                result.setValue(i, j, (double)1.0F / ((double)1.0F + Math.exp(-(Double)matrix.getValue(i, j))));
            }
        }
        return result;
    }

    @Override
    public Matrix derivative(Matrix matrix) {
        Matrix result = new Matrix(matrix.getRow(), matrix.getColumn());
        for (int i = 0; i < matrix.getRow(); i++) {
            for (int j = 0; j < matrix.getColumn(); j++) {
                result.setValue(i, j, matrix.getValue(i, j) * (1.0F - matrix.getValue(i, j)));
            }
        }
        return result;
    }
}
