/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.commons.math3.optim.linear;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.math3.exception.TooManyIterationsException;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.util.Precision;

/**
 * Solves a linear problem using the "Two-Phase Simplex" method.
 * <p>
 * <b>Note:</b> Depending on the problem definition, the default convergence criteria
 * may be too strict, resulting in {@link NoFeasibleSolutionException} or
 * {@link TooManyIterationsException}. In such a case it is advised to adjust these
 * criteria with more appropriate values, e.g. relaxing the epsilon value.
 * <p>
 * Default convergence criteria:
 * <ul>
 *   <li>Algorithm convergence: 1e-6</li>
 *   <li>Floating-point comparisons: 10 ulp</li>
 *   <li>Cut-Off value: 1e-12</li>
 * </ul>
 * <p>
 * The cut-off value has been introduced to zero out very small numbers in the Simplex tableau,
 * as these may lead to numerical instabilities due to the nature of the Simplex algorithm
 * (the pivot element is used as a denominator). If the problem definition is very tight, the
 * default cut-off value may be too small, thus it is advised to increase it to a larger value,
 * in accordance with the chosen epsilon.
 * <p>
 * It may also be counter-productive to provide a too large value for {@link
 * org.apache.commons.math3.optim.MaxIter MaxIter} as parameter in the call of {@link
 * #optimize(org.apache.commons.math3.optim.OptimizationData...) optimize(OptimizationData...)},
 * as the {@link SimplexSolver} will use different strategies depending on the current iteration
 * count. After half of the allowed max iterations has already been reached, the strategy to select
 * pivot rows will change in order to break possible cycles due to degenerate problems.
 *
 * @version $Id$
 * @since 2.0
 */
public class SimplexSolver extends LinearOptimizer {
    /** Default amount of error to accept in floating point comparisons (as ulps). */
    static final int DEFAULT_ULPS = 10;

    /** Default cut-off value. */
    static final double DEFAULT_CUT_OFF = 1e-12;

    /** Default amount of error to accept for algorithm convergence. */
    private static final double DEFAULT_EPSILON = 1.0e-6;

    /** Amount of error to accept for algorithm convergence. */
    private final double epsilon;

    /** Amount of error to accept in floating point comparisons (as ulps). */
    private final int maxUlps;

    /**
     * Cut-off value for entries in the tableau: values smaller than the cut-off
     * are treated as zero to improve numerical stability.
     */
    private final double cutOff;

    /**
     * The solution callback to access the best solution found so far in case
     * the optimizer fails to find an optimal solution within the iteration limits.
     */
    private SolutionCallback solutionCallback;

    /**
     * Builds a simplex solver with default settings.
     */
    public SimplexSolver() {
        this(DEFAULT_EPSILON, DEFAULT_ULPS, DEFAULT_CUT_OFF);
    }

    /**
     * Builds a simplex solver with a specified accepted amount of error.
     *
     * @param epsilon Amount of error to accept for algorithm convergence.
     */
    public SimplexSolver(final double epsilon) {
        this(epsilon, DEFAULT_ULPS, DEFAULT_CUT_OFF);
    }

    /**
     * Builds a simplex solver with a specified accepted amount of error.
     *
     * @param epsilon Amount of error to accept for algorithm convergence.
     * @param maxUlps Amount of error to accept in floating point comparisons.
     */
    public SimplexSolver(final double epsilon, final int maxUlps) {
        this(epsilon, maxUlps, DEFAULT_CUT_OFF);
    }

    /**
     * Builds a simplex solver with a specified accepted amount of error.
     *
     * @param epsilon Amount of error to accept for algorithm convergence.
     * @param maxUlps Amount of error to accept in floating point comparisons.
     * @param cutOff Values smaller than the cutOff are treated as zero.
     */
    public SimplexSolver(final double epsilon, final int maxUlps, final double cutOff) {
        this.epsilon = epsilon;
        this.maxUlps = maxUlps;
        this.cutOff = cutOff;
    }

    /**
     * {@inheritDoc}
     *
     * @param optData Optimization data. In addition to those documented in
     * {@link LinearOptimizer#optimize(OptimizationData...)
     * LinearOptimizer}, this method will register the following data:
     * <ul>
     *  <li>{@link SolutionCallback}</li>
     * </ul>
     *
     * @return {@inheritDoc}
     * @throws TooManyIterationsException if the maximal number of iterations is exceeded.
     */
    @Override
    public PointValuePair optimize(OptimizationData... optData)
        throws TooManyIterationsException {
        // Set up base class and perform computation.
        return super.optimize(optData);
    }

    /**
     * {@inheritDoc}
     *
     * @param optData Optimization data.
     * In addition to those documented in
     * {@link LinearOptimizer#parseOptimizationData(OptimizationData[])
     * LinearOptimizer}, this method will register the following data:
     * <ul>
     *  <li>{@link SolutionCallback}</li>
     * </ul>
     */
    @Override
    protected void parseOptimizationData(OptimizationData... optData) {
        // Allow base class to register its own data.
        super.parseOptimizationData(optData);

        // reset the callback before parsing
        solutionCallback = null;

        for (OptimizationData data : optData) {
            if (data instanceof SolutionCallback) {
                solutionCallback = (SolutionCallback) data;
                continue;
            }
        }
    }

    /**
     * Returns the column with the most negative coefficient in the objective function row.
     *
     * @param tableau Simple tableau for the problem.
     * @return the column with the most negative coefficient.
     */
    private Integer getPivotColumn(SimplexTableau tableau) {
        double minValue = 0;
        Integer minPos = null;
        for (int i = tableau.getNumObjectiveFunctions(); i < tableau.getWidth() - 1; i++) {
            final double entry = tableau.getEntry(0, i);
            // check if the entry is strictly smaller than the current minimum
            // do not use a ulp/epsilon check
            if (entry < minValue) {
                minValue = entry;
                minPos = i;
            }
        }
        return minPos;
    }

    /**
     * Returns the row with the minimum ratio as given by the minimum ratio test (MRT).
     *
     * @param tableau Simple tableau for the problem.
     * @param col Column to test the ratio of (see {@link #getPivotColumn(SimplexTableau)}).
     * @return the row with the minimum ratio.
     */
    private Integer getPivotRow(SimplexTableau tableau, final int col) {
        // create a list of all the rows that tie for the lowest score in the minimum ratio test
        List<Integer> minRatioPositions = new ArrayList<Integer>();
        double minRatio = Double.MAX_VALUE;
        for (int i = tableau.getNumObjectiveFunctions(); i < tableau.getHeight(); i++) {
            final double rhs = tableau.getEntry(i, tableau.getWidth() - 1);
            final double entry = tableau.getEntry(i, col);

            if (Precision.compareTo(entry, 0d, maxUlps) > 0) {
                final double ratio = rhs / entry;
                // check if the entry is strictly equal to the current min ratio
                // do not use a ulp/epsilon check
                final int cmp = Double.compare(ratio, minRatio);
                if (cmp == 0) {
                    minRatioPositions.add(i);
                } else if (cmp < 0) {
                    minRatio = ratio;
                    minRatioPositions = new ArrayList<Integer>();
                    minRatioPositions.add(i);
                }
            }
        }

        if (minRatioPositions.size() == 0) {
            return null;
        } else if (minRatioPositions.size() > 1) {
            // there's a degeneracy as indicated by a tie in the minimum ratio test

            // 1. check if there's an artificial variable that can be forced out of the basis
            if (tableau.getNumArtificialVariables() > 0) {
                for (Integer row : minRatioPositions) {
                    for (int i = 0; i < tableau.getNumArtificialVariables(); i++) {
                        int column = i + tableau.getArtificialVariableOffset();
                        final double entry = tableau.getEntry(row, column);
                        if (Precision.equals(entry, 1d, maxUlps) && row.equals(tableau.getBasicRow(column))) {
                            return row;
                        }
                    }
                }
            }

            // 2. apply Bland's rule to prevent cycling:
            //    take the row for which the corresponding basic variable has the smallest index
            //
            // see http://www.stanford.edu/class/msande310/blandrule.pdf
            // see http://en.wikipedia.org/wiki/Bland%27s_rule (not equivalent to the above paper)
            //
            // Additional heuristic: if we did not get a solution after half of maxIterations
            //                       revert to the simple case of just returning the top-most row
            // This heuristic is based on empirical data gathered while investigating MATH-828.
            if (getEvaluations() < getMaxEvaluations() / 2) {
                Integer minRow = null;
                int minIndex = tableau.getWidth();
                final int varStart = tableau.getNumObjectiveFunctions();
                final int varEnd = tableau.getWidth() - 1;
                for (Integer row : minRatioPositions) {
                    for (int i = varStart; i < varEnd && !row.equals(minRow); i++) {
                        final Integer basicRow = tableau.getBasicRow(i);
                        if (basicRow != null && basicRow.equals(row) && i < minIndex) {
                            minIndex = i;
                            minRow = row;
                        }
                    }
                }
                return minRow;
            }
        }
        return minRatioPositions.get(0);
    }

    /**
     * Runs one iteration of the Simplex method on the given model.
     *
     * @param tableau Simple tableau for the problem.
     * @throws TooManyIterationsException if the allowed number of iterations has been exhausted.
     * @throws UnboundedSolutionException if the model is found not to have a bounded solution.
     */
    protected void doIteration(final SimplexTableau tableau)
        throws TooManyIterationsException,
               UnboundedSolutionException {

        incrementIterationCount();

        Integer pivotCol = getPivotColumn(tableau);
        Integer pivotRow = getPivotRow(tableau, pivotCol);
        if (pivotRow == null) {
            throw new UnboundedSolutionException();
        }

        // set the pivot element to 1
        double pivotVal = tableau.getEntry(pivotRow, pivotCol);
        tableau.divideRow(pivotRow, pivotVal);

        // set the rest of the pivot column to 0
        for (int i = 0; i < tableau.getHeight(); i++) {
            if (i != pivotRow) {
                final double multiplier = tableau.getEntry(i, pivotCol);
                tableau.subtractRow(i, pivotRow, multiplier);
            }
        }
    }

    /**
     * Solves Phase 1 of the Simplex method.
     *
     * @param tableau Simple tableau for the problem.
     * @throws TooManyIterationsException if the allowed number of iterations has been exhausted.
     * @throws UnboundedSolutionException if the model is found not to have a bounded solution.
     * @throws NoFeasibleSolutionException if there is no feasible solution?
     */
    protected void solvePhase1(final SimplexTableau tableau)
        throws TooManyIterationsException,
               UnboundedSolutionException,
               NoFeasibleSolutionException {

        // make sure we're in Phase 1
        if (tableau.getNumArtificialVariables() == 0) {
            return;
        }

        while (!tableau.isOptimal()) {
            doIteration(tableau);
        }

        // if W is not zero then we have no feasible solution
        if (!Precision.equals(tableau.getEntry(0, tableau.getRhsOffset()), 0d, epsilon)) {
            throw new NoFeasibleSolutionException();
        }
    }

    /** {@inheritDoc} */
    @Override
    public PointValuePair doOptimize()
        throws TooManyIterationsException,
               UnboundedSolutionException,
               NoFeasibleSolutionException {
        SimplexTableau tableau = getOptimalTableau();

        return tableau.getSolution();
    }
    
    public SimplexSolution doOptimizeExtra(OptimizationData... data)
            throws TooManyIterationsException,
            UnboundedSolutionException,
            NoFeasibleSolutionException {
        
        parseOptimizationData(data);
        
        SimplexTableau tableau = getOptimalTableau();
        
        return new SimplexSolution(tableau.getSolution(), getExtraConstraintInfo(tableau), getCoefficientSensitivity(tableau));
    }
    
    private SimplexTableau getOptimalTableau()
    {
        // reset the tableau to indicate a non-feasible solution in case
        // we do not pass phase 1 successfully
        if (solutionCallback != null) {
            solutionCallback.setTableau(null);
        }

        final SimplexTableau tableau =
            new SimplexTableau(getFunction(),
                               getConstraints(),
                               getGoalType(),
                               isRestrictedToNonNegative(),
                               epsilon,
                               maxUlps,
                               cutOff);

        solvePhase1(tableau);
        tableau.dropPhase1Objective();

        // after phase 1, we are sure to have a feasible solution
        if (solutionCallback != null) {
            solutionCallback.setTableau(tableau);
        }

        while (!tableau.isOptimal()) {
            doIteration(tableau);
        }
        
        return tableau;
    }
    
    private HashMap<LinearConstraint, ExtraConstraintInfo> getExtraConstraintInfo(SimplexTableau tableau) {
        
        HashMap<LinearConstraint, ExtraConstraintInfo> constraintMap = new HashMap<LinearConstraint, ExtraConstraintInfo>();
        
        Double shadowPrice;
        Double lowerBound;
        Double upperBound;
        
        int slackVariableCount = 0;
        List<LinearConstraint> originalConstraints = tableau.getOriginalConstraints();
        
        for (int constraintIndex = 0; constraintIndex < originalConstraints.size(); constraintIndex++)
        {
            LinearConstraint constraint = originalConstraints.get(constraintIndex);
                    
            if (constraint.getRelationship() == Relationship.EQ)
            {
                continue;
            }
            
            int column = tableau.getSlackVariableOffset() + slackVariableCount;
            slackVariableCount++;
            
            shadowPrice = -1d * tableau.getEntry(0, column);
            
            Integer row = tableau.getBasicRow(column);
            
            if (row != null) // Basic variable
            {
                Double rhs = tableau.getEntry(row, tableau.getRhsOffset());
                lowerBound = -rhs;
                upperBound = rhs;
            }
            else
            {
                Double[] bounds = findBounds(tableau, column, (constraint.getRelationship() == Relationship.LEQ) ? -1d : 1d);
                lowerBound = bounds[0];
                upperBound = bounds[1];
            }
            
            ExtraConstraintInfo extraConstraintInfo = new ExtraConstraintInfo(shadowPrice, 
                    lowerBound == Double.NEGATIVE_INFINITY ? null : (constraint.getValue() + lowerBound), 
                    upperBound == Double.POSITIVE_INFINITY ? null : (constraint.getValue() + upperBound));
            
            constraintMap.put (constraint, extraConstraintInfo);
        }        
        
        return constraintMap;
    }
    
    private CoefficientBounds[] getCoefficientSensitivity(SimplexTableau tableau) {
        
        double[] coefficients = getFunction().getCoefficients().toArray();
        
        int numObjectiveFunctions = tableau.getNumObjectiveFunctions();
        
        int originalNumDecisionVariables = tableau.getOriginalNumDecisionVariables();
        
        CoefficientBounds[] ret = new CoefficientBounds[originalNumDecisionVariables];
        
        for (int i = 0; i < originalNumDecisionVariables; i++)
        {
            int column = numObjectiveFunctions + i;
            
            Integer k = tableau.getBasicRow(column);
            
            if (k == null)
            {
                double reducedCost = tableau.getEntry(0, column);
                
                ret[i] = new CoefficientBounds(coefficients[i], null, coefficients[i] - reducedCost);
            }
            else
            {
                double max = Double.NEGATIVE_INFINITY;
                double min = Double.POSITIVE_INFINITY;

                for (int j = 0; j < tableau.getRhsOffset(); j++)
                {
                    double akj = tableau.getEntry(k, numObjectiveFunctions + j);

                    if (Precision.equals(0d, akj, epsilon)) 
                       continue;

                    double ratio = tableau.getEntry(0, numObjectiveFunctions + j) / akj;

                    if (akj > 0d)
                    {
                        max = Math.max (max, ratio);
                    }
                    else
                    {
                        min = Math.min (min, ratio);
                    }
                }

                ret[i] = new CoefficientBounds(coefficients[i], 
                            min == Double.POSITIVE_INFINITY ? null : (coefficients[i] + min),
                            max == Double.NEGATIVE_INFINITY ? null : (coefficients[i] + max));

            }
        }
        
        return ret;        
    }
    
    private Double[] findBounds(SimplexTableau tableau, int column, double multiplier) {
        
        Double min = Double.NEGATIVE_INFINITY;
        Double max = Double.POSITIVE_INFINITY;
        
        for (int row = tableau.getNumObjectiveFunctions(); row < tableau.getHeight(); row++)
        {
            double bi = tableau.getEntry(row, tableau.getRhsOffset());
            
            if (Precision.equals(bi, 0d, epsilon))
                continue;
            
            double betaik = tableau.getEntry(row, column);
            
            if (Precision.equals(0d, betaik, epsilon) || 
                Precision.equals(1d, betaik, epsilon))
                continue;
            
            double ratio = multiplier * (bi / betaik);
            
            if (ratio < 0) // betaik > 0
            {
                min = Math.max(min, ratio);
            }
            else
            {
                max = Math.min(max, ratio);
            }
        }
        
        return new Double[] { min, max};
    }
}
