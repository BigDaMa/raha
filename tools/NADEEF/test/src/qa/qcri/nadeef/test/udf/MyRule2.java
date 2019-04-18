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

import com.google.common.collect.Lists;
import qa.qcri.nadeef.core.datamodel.*;

import java.util.Collection;

/**
 * My custom rule with pair detection.
 */
public class MyRule2 extends PairTupleRule {
    /**
     * Detect rule with one tuple.
     *
     * @param tuples input tuple.
     * @return Violation set.
     */
    @Override
    public Collection<Violation> detect(TuplePair tuples) {
        Tuple left = tuples.getLeft();
        Tuple right = tuples.getRight();

        Column city = new Column(getTableNames().get(0), "city");

        Violation violation = new Violation(getRuleName());
        if (!left.get(city).equals(right.get(city))) {
            violation.addTuple(left);
			violation.addTuple(right);
        }

        return Lists.newArrayList(violation);
    }

    /**
     * Repair of this rule.
     *
     * @param violation violation input.
     * @return a candidate fix.
     */
    @Override
    public Collection<Fix> repair(Violation violation) {
        return null;
    }

    @Override
    public Collection<Table> horizontalScope(
        Collection<Table> tables
    ) {
        Table collection = tables.iterator().next();
        String tableName = getTableNames().get(0);
        collection.filter(
            new Predicate.PredicateBuilder()
                .left(new Column(tableName, "zip"))
                .isSingle()
                .constant("1183JV")
                .op(Operation.EQ)
                .build()
        );
        return tables;
    }

    @Override
    public Collection<Table> verticalScope(
        Collection<Table> tables
    ) {
        Table collection = tables.iterator().next();
        collection.project("city").project("zip");
        return tables;
    }
}
