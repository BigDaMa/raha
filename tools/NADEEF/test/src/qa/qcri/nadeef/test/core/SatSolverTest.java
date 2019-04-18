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

package qa.qcri.nadeef.test.core;

import com.google.common.collect.Lists;
import org.junit.Assert;
import org.junit.Test;
import org.sat4j.core.VecInt;
import org.sat4j.maxsat.SolverFactory;
import org.sat4j.specs.IProblem;
import org.sat4j.specs.ISolver;
import qa.qcri.nadeef.core.datamodel.Cell;
import qa.qcri.nadeef.core.datamodel.Column;
import qa.qcri.nadeef.core.datamodel.Fix;
import qa.qcri.nadeef.core.datamodel.Operation;
import qa.qcri.nadeef.core.pipeline.SatSolver;

import java.util.List;

public class SatSolverTest {
    @Test
    public void testSat4j() {
        ISolver solver = SolverFactory.newDefault();
        solver.newVar(10);
        solver.setExpectedNumberOfClauses(100);
        try {
            solver.addClause(new VecInt(new int[] { 1, 2, 3 }));
            solver.addClause(new VecInt(new int[] { -1 }));
            solver.addClause(new VecInt(new int[] { -2 }));
            solver.addClause(new VecInt(new int[] { 2, 3 }));

            IProblem problem = solver;
            boolean isSat = problem.isSatisfiable();
            Assert.assertTrue(isSat);
            if (isSat) {
                int[] model = problem.model();
                Assert.assertEquals(3, model.length);
                Assert.assertTrue(model[0] > 0 || model[1] > 0 || model[2] > 0);
                Assert.assertTrue(model[0] < 0);
                Assert.assertTrue(model[1] < 0);
                Assert.assertTrue(model[2] > 0);
            }

        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    @Test
    public void testSatSolver1() {
        List<Fix> fixes = Lists.newArrayList();
        Fix.Builder builder = new Fix.Builder();
        Column column = new Column("dummy", "testColumn");
        Cell cellA = new Cell(column, 1, "A");
        Cell cellB = new Cell(column, 2, "B");
        Cell cellC = new Cell(column, 3, "C");
        Fix a = builder.left(cellA).op(Operation.EQ).right(cellB).vid(1).build();
        Fix b = builder.left(cellB).op(Operation.EQ).right(cellC).vid(2).build();
        Fix c = builder.left(cellA).op(Operation.EQ).right("D").vid(3).build();
        fixes.add(a);
        fixes.add(b);
        fixes.add(c);

        SatSolver solver = new SatSolver(null);
        List<Fix> result = Lists.newArrayList(solver.decide(fixes));
        Assert.assertEquals(3, result.size());
    }

}
