package org.apache.commons.math3.optim.linear;

/**
 *
 * @author Renato Cherullo <renato.cherullo@rerum.com.br>
 */
public class ExtraConstraintInfo {

    private final Double shadowPrice;
    
    private final Double lowerBound;
    
    private final Double upperBound;

    public ExtraConstraintInfo(Double shadowPrice, Double lowerBound, Double upperBound) {
        this.shadowPrice = shadowPrice;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
    }

    public Double getShadowPrice() {
        return shadowPrice;
    }

    public Double getLowerBound() {
        return lowerBound;
    }

    public Double getUpperBound() {
        return upperBound;
    }

}
