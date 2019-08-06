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
import qa.qcri.nadeef.tools.Metrics;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class DedupRule extends PairTupleRule {
    @Override
    public void initialize(String id, List<String> tableNames) {
        super.initialize(id, tableNames);
    }

    @Override
    public Collection<Table> block(Collection<Table> tables) {
        Table table = tables.iterator().next();
        return table.groupOn("brand_name");
    }

    @Override
    public Collection<Violation> detect(TuplePair tuplePair) {
        List<Violation> result = new ArrayList<>();
        Tuple left = tuplePair.getLeft();
        Tuple right = tuplePair.getRight();

        if (true
            && Metrics.getEqual(getValue(tuplePair, "qcars", "model", 0), getValue(tuplePair, "qcars", "model", 1))==1
            && Metrics.getEuclideanDistance(getValue(tuplePair, "qcars", "contact_number", 0),getValue(tuplePair, "qcars", "contact_number", 1))>0.8
            && Metrics.getEqual(getValue(tuplePair, "qcars", "brand_name", 0),getValue(tuplePair, "qcars", "brand_name", 1))==1
        ) {
            Violation violation = new Violation(getRuleName());
            violation.addTuple(left);
            violation.addTuple(right);
            result.add(violation);
        }

        return result;
    }

    @Override
    public Collection<Fix> repair(Violation violation) {
        return new ArrayList<>();
    }

    private String getValue(TuplePair pair, String tableName, String column, int isLeft) {
        Tuple left = pair.getLeft();
        Tuple right = pair.getRight();
        String result;
        if (isLeft == 0) {
            if (left.isFromTable(tableName)) {
                Object obj = left.get(column);
                result = obj != null ? obj.toString() : null;
            } else {
                Object obj = right.get(column);
                result = obj != null ? obj.toString() : null;
            }
        } else {
            if (right.isFromTable(tableName)) {
                Object obj = right.get(column);
                result = obj != null ? obj.toString() : null;
            } else {
                Object obj = left.get(column);
                result = obj != null ? obj.toString() : null;
            }
        }
        return result;
    }
}