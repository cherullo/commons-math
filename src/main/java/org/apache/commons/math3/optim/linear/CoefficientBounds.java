package org.apache.commons.math3.optim.linear;

/**
 *
 * @author Renato Cherullo <renato.cherullo@rerum.com.br>
 */
public class CoefficientBounds {

    private final Double objectiveFunctionCoefficient;
    private final Double lowerBound;
    private final Double upperBound;

    public CoefficientBounds(Double objectiveFunctionCoefficient, Double lowerBound, Double upperBound) {
        this.objectiveFunctionCoefficient = objectiveFunctionCoefficient;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
    }

    public Double getObjectiveFunctionCoefficient() {
        return objectiveFunctionCoefficient;
    }

    public Double getLowerBound() {
        return lowerBound;
    }

    public Double getUpperBound() {
        return upperBound;
    }
}
