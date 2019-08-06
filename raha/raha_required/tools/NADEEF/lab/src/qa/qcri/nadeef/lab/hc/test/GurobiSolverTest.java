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

import org.junit.Assert;
import org.junit.Test;
import qa.qcri.nadeef.core.datamodel.Cell;
import qa.qcri.nadeef.core.datamodel.Column;
import qa.qcri.nadeef.core.datamodel.Fix;
import qa.qcri.nadeef.core.datamodel.Operation;
import qa.qcri.nadeef.lab.hc.GurobiSolver;
import qa.qcri.nadeef.lab.hc.HolisticCleaning;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class GurobiSolverTest {
    @Test
    public void simpleTest1() {
        // We are testing
        //      t.a > t.b, t.b > t.c
        //      t.a has value 2
        //      t.b has value 4
        //      t.c has value 6
        //      t.a > 3
        //      t.b != 5
        HashSet<Fix> fixSet = new HashSet<Fix>();
        Fix.Builder builder = new Fix.Builder();
        Cell ta = new Cell(new Column("T", "A"), 1, 2.0);
        Cell tb = new Cell(new Column("T", "B"), 1, 4.0);
        Cell tc = new Cell(new Column("T", "C"), 1, 6.0);
        fixSet.add(builder.left(ta).right(tb).op(Operation.GT).build());
        fixSet.add(builder.left(tb).right(tc).op(Operation.GT).build());
        fixSet.add(builder.left(ta).right(3).op(Operation.GT).build());
        // fixSet.add(builder.left(tb).right(5).op(Operation.NEQ).build());

        List<Fix> result = new GurobiSolver().solve(fixSet);
        double va = ta.getValue(), vb = tb.getValue(), vc = tc.getValue();
        for (Fix fix : result) {
            if (fix.getLeft().equals(ta))
                va = Math.round(Double.parseDouble(fix.getRightValue()));

            if (fix.getLeft().equals(tb))
                vb = Math.round(Double.parseDouble(fix.getRightValue()));

            if (fix.getLeft().equals(tc))
                vc = Math.round(Double.parseDouble(fix.getRightValue()));
        }

        Assert.assertTrue(va > vb);
        Assert.assertTrue(vb > vc);
        Assert.assertTrue(va > 3);
        Assert.assertTrue(vb != 5);
    }

    @Test
    public void simpleTest2() {
        // We are testing
        //      t.a < t.b, t.b < t.c
        //      t.a has value 0
        //      t.b has value 3
        //      t.c has value 2
        HashSet<Fix> fixSet = new HashSet<>();
        Fix.Builder builder = new Fix.Builder();
        Cell ta = new Cell(new Column("T", "A"), 1, 0.0);
        Cell tb = new Cell(new Column("T", "B"), 1, 3.0);
        Cell tc = new Cell(new Column("T", "C"), 1, 2.0);
        fixSet.add(builder.left(ta).right(tb).op(Operation.LT).build());
        fixSet.add(builder.left(tb).right(tc).op(Operation.LT).build());

        List<Fix> result = new ArrayList<>(new HolisticCleaning(null).decide(fixSet));
        double va = ta.getValue(), vb = tb.getValue(), vc = tc.getValue();
        for (Fix fix : result) {
            if (fix.getLeft().equals(ta))
                va = Double.parseDouble(fix.getRightValue());

            if (fix.getLeft().equals(tb))
                vb = Double.parseDouble(fix.getRightValue());

            if (fix.getLeft().equals(tc))
                vc = Double.parseDouble(fix.getRightValue());
        }

        Assert.assertTrue(va < vb);
        Assert.assertTrue(vb < vc);
        Assert.assertTrue(vb == 1.0);
    }

    @Test
    public void infeasibleTest() {
        // We are testing
        //      t.a > t.b, t.a < t.c
        //      t.a has value 2
        //      t.b has value 4
        //      t.c has value 6
        //      t.b > 3
        //      t.c < 2
        HashSet<Fix> fixSet = new HashSet<Fix>();
        Fix.Builder builder = new Fix.Builder();
        Cell ta = new Cell(new Column("T", "A"), 1, 2.0);
        Cell tb = new Cell(new Column("T", "B"), 1, 4.0);
        Cell tc = new Cell(new Column("T", "C"), 1, 6.0);
        fixSet.add(builder.left(ta).right(tb).op(Operation.GT).build());
        fixSet.add(builder.left(ta).right(tc).op(Operation.LT).build());
        fixSet.add(builder.left(tb).right(3).op(Operation.GT).build());
        fixSet.add(builder.left(tc).right(5).op(Operation.LT).build());

        List<Fix> result = new GurobiSolver().solve(fixSet);
        Assert.assertEquals(result, null);
        /*
        double va = ta.getValue(), vb = tb.getValue(), vc = tc.getValue();
        for (Fix fix : result) {
            if (fix.getLeft().equals(ta))
                va = Double.parseDouble(fix.getRightValue());

            if (fix.getLeft().equals(tb))
                vb = Double.parseDouble(fix.getRightValue());

            if (fix.getLeft().equals(tc))
                vc = Double.parseDouble(fix.getRightValue());
        }

        Assert.assertTrue(va > vb);
        Assert.assertTrue(va < vc);
        */
    }
}
