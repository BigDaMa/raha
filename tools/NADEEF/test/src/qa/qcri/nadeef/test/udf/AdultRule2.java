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

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import qa.qcri.nadeef.core.datamodel.*;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Test rule.
 */
public class AdultRule2 extends PairTupleRule {
    @Override
    public Collection<Table> horizontalScope(
        Collection<Table> tables
    ) {
        Table table = tables.iterator().next();
        table.project("race").project("fnlwgt");
        return tables;
    }

    @Override
    public Collection<Table> block(Collection<Table> tables) {
        Table table = Iterables.get(tables, 0);
        return table.groupOn("race");
    }

    @Override
    public void iterator(Collection<Table> tables, IteratorResultHandler iteratorResultHandler) {
        Table table = tables.iterator().next();
        ArrayList<TuplePair> result = new ArrayList<>();
        for (int i = 0; i < table.size(); i ++) {
            for (int j = i + 1; j < table.size(); j ++) {
                Tuple left = table.get(i);
                Tuple right = table.get(j);
                if (!left.get("fnlwgt").equals(right.get("fnlwgt"))) {
                    TuplePair pair = new TuplePair(table.get(i), table.get(j));
                    iteratorResultHandler.handle(pair);
                    break;
                }
            }
        }
    }

    @Override
    public Collection<Violation> detect(TuplePair tuplePair) {
        List<Violation> result = Lists.newArrayList();

        Tuple t1 = tuplePair.getLeft();
        Tuple t2 = tuplePair.getRight();
        if (t1.get("race").equals(t2.get("race"))) {
            int wgt1 = (Integer)t1.get("fnlwgt");
            int wgt2 = (Integer)t2.get("fnlwgt");
            if (Math.abs(wgt1 - wgt2) <= wgt1 * 0.5) {
                Violation violation = new Violation(getRuleName());
                violation.addCell(t1.getCell("race"));
                violation.addCell(t2.getCell("race"));
                result.add(violation);
            }
        }
        return result;
    }

    @Override
    public Collection<Fix> repair(Violation violation) {
        List<Fix> result = Lists.newArrayList();
        return result;
    }
}
