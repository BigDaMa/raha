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

package qa.qcri.nadeef.test.udf;

import qa.qcri.nadeef.core.datamodel.*;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

/**
 * Single Tuple CFD rule.
 *
 */
public class AdultRule3 extends SingleTupleRule {
    protected List<Column> lhs;
    protected List<Column> rhs;
    protected List<Predicate> leftFilterExpressions;
    protected List<Predicate> rightFilterExpressions;
    protected HashMap<Column, Predicate> filterCache;

    @Override
    public void initialize(String id, List<String> tableNames) {
        super.initialize(id, tableNames);

        lhs = new ArrayList<>();
        rhs = new ArrayList<>();
        leftFilterExpressions = new ArrayList<>();
        rightFilterExpressions = new ArrayList<>();
        filterCache = new HashMap<>();

        lhs.add(new Column("TB_ADULT_1K.relationship"));

        rhs.add(new Column("TB_ADULT_1K.sex"));

        leftFilterExpressions.add(
            Predicate.createEq(new Column("TB_ADULT_1K.relationship"), "Wife"));

        rightFilterExpressions.add(
            Predicate.createEq(new Column("TB_ADULT_1K.sex"), "Female"));
    }

    /**
     * Default horizontal scope operation.
     * @param tables input tables.
     * @return filtered tables.
     */
    @Override
    public Collection<Table> horizontalScope(Collection<Table> tables) {
        tables.iterator().next().project(lhs).project(rhs);
        return tables;
    }

    /**
     * Default vertical scope operation.
     * @param tables input tables.
     * @return filtered tables.
     */
    @Override
    public Collection<Table> verticalScope(Collection<Table> tables) {
        tables.iterator().next().filter(leftFilterExpressions);
        return tables;
    }

    /**
     * Default iterator operation.
     *
     * @param blocks input table
     */
    @Override
    public void iterator(Collection<Table> blocks, IteratorResultHandler output) {
        Table table = blocks.iterator().next();
        ArrayList<Tuple> result = new ArrayList<>();
        int pos = 0;
        while (pos < table.size()) {
            Tuple t = table.get(pos++);
            output.handle(t);
        }
    }

    /**
     * Detect rule with one tuple.
     *
     * @param tuple input tuple.
     * @return Violation collection.
     */
    @Override
    public Collection<Violation> detect(Tuple tuple) {
        List<Violation> result = new ArrayList<>();
        // Matching with lhs is redundant, but needed for correctness of detect function,
        // when used in incremental detection.

        if(matches(leftFilterExpressions, tuple)) {
            if(!matches(rightFilterExpressions, tuple)) {
                Violation violation = new Violation(getRuleName());
                violation.addTuple(tuple);
                result.add(violation);
            }
        }
        return result;
    }

    /**
     * Repair of this rule.
     *
     * @param violation violation input.
     * @return a candidate fix.
     */
    @Override
    public Collection<Fix> repair(Violation violation) {
        return new ArrayList<>();
    }

    private static boolean matches(List<Predicate> expressions, Tuple tuple) {
        for (Predicate exp : expressions) {
            if (!exp.isValid(tuple)) {
                return false;
            }
        }
        return true;
    }
}