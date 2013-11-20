package org.apache.commons.math3.optim.linear;

import java.util.HashMap;
import org.apache.commons.math3.optim.PointValuePair;

/**
 *
 * @author Renato Cherullo <renato.cherullo@rerum.com.br>
 */
public class SimplexSolution {

    private PointValuePair solution;
    private HashMap<LinearConstraint, ExtraConstraintInfo> extraConstraintInfo;
    private CoefficientBounds[] sensitivity;

    public SimplexSolution(PointValuePair solution, HashMap<LinearConstraint, ExtraConstraintInfo> extraConstraintInfo, CoefficientBounds[] sensitivity) {
        this.solution = solution;
        this.extraConstraintInfo = extraConstraintInfo;
        this.sensitivity = sensitivity;
    }

    public PointValuePair getSolution() {
        return solution;
    }

    public HashMap<LinearConstraint, ExtraConstraintInfo> getExtraConstraintInfo() {
        return extraConstraintInfo;
    }

    public CoefficientBounds[] getSensitivity() {
        return sensitivity;
    }
        
}
