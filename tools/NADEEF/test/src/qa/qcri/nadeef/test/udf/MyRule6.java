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
import java.util.List;

/**
 */
public class MyRule6 extends SingleTupleRule {
    /**
     * Detect rule with one tuple.
     *
     * @param tuple input tuple.
     * @return Violation set.
     */
    @Override
    public Collection<Violation> detect(Tuple tuple) {
        List<Violation> result = Lists.newArrayList();
        if (tuple.getCell("CC").getValue().equals("31") &&
            (
                !tuple.getCell("country").getValue().equals("Netherlands") ||
                !tuple.getCell("country").getValue().equals("Holland"))
            )
        {
            Violation violation = new Violation(getRuleName());
            violation.addCell(tuple.getCell("cc"));
            violation.addCell(tuple.getCell("country"));
            result.add(violation);
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
        Fix.Builder builder = new Fix.Builder();
        Cell countryCell = violation.getCell("tran1", "country");
        Fix fix = builder.left(countryCell).right("Netherlands").build();
        return Lists.newArrayList(fix);
    }
}
