/*
 * QCRI, NADEEF LICENSE
 * NADEEF is an extensible, generalized and easy-to-deploy data cleaning platform built at QCRI.
 * NADEEF means "Clean" in Arabic
 *
 * Copyright (c) 2011-2013, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
 * NADEEF has patent pending nevertheless the following is granted.
 * NADEEF is released under the terms of the MIT License, (http://opensource.org/licenses/MIT).
 */

package qa.qcri.nadeef.lab.hc.test;

import org.junit.BeforeClass;
import org.junit.Test;

public class JOptimizerTest {
    @BeforeClass
    public static void beforeClass() {
    }

    @Test
    public void simpleQpTest() {
        // minimize (x - 1)^2 + (y - 2)^2
        // with constraint that
        //      x > y
        //      y > 0
        //      3x = 1
        //
        /*
        double[][] P = new double[][] { { 1.0, 0.0 }, { 0.0, 1.0 }};
        double[] Q = new double[] { -2.0, -4.0 };
        double r = 5.0;
        PDQuadraticMultivariateRealFunction objFunc =
            new PDQuadraticMultivariateRealFunction(P, Q, r);

        ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[1];
        inequalities[0] = new LinearMultivariateRealFunction(new double[] { -1.0, 1.0 }, 0.0);

        OptimizationRequest or = new OptimizationRequest();
        or.setF0(objFunc);
        or.setA(new double[][] {{ 3.0, 0.0 }});
        or.setB(new double[] { 1.0 });
        or.setInitialPoint(new double[] { 0.3, 0.1 });
        or.setFi(inequalities);
        or.setToleranceFeas(1.0);
        or.setTolerance(10.0);

        try {
            JOptimizer opt = new JOptimizer();
            opt.setOptimizationRequest(or);
            int returnCode = opt.optimize();
            double[] solution = opt.getOptimizationResponse().getSolution();
            System.out.println(solution[0]);
            System.out.println(solution[1]);
            Assert.assertEquals(2, solution.length);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        */
    }
}
