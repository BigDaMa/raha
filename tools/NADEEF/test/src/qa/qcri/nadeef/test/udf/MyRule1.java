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

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Test rule.
 */
public class MyRule1 extends SingleTupleRule {
    /**
     * Detect rule with one tuple.
     *
     * @param tuple input tuple.
     * @return Violation set.
     */
    @Override
    public Collection<Violation> detect(Tuple tuple) {
        List<Violation> result = new ArrayList<>();
        String city = (String)tuple.get("city");
        String zip = (String)tuple.get("zip");
        if (zip.equalsIgnoreCase("1183JV")) {
            if (!city.equalsIgnoreCase("amsterdam")) {
                Violation newViolation = new Violation(getRuleName());
                newViolation.addCell(tuple.getCell("city"));
                newViolation.addCell(tuple.getCell("zip"));
                result.add(newViolation);
            }
        }
        return result;
    }

    /**
     * Repair of this rule.
     *
     *
     * @param violation violation input.
     * @return a candidate fix.
     */
    @Override
    public Collection<Fix> repair(Violation violation) {
        List<Fix> result = Lists.newArrayList();
        List<Cell> cells = Lists.newArrayList(violation.getCells());
        Fix.Builder fixBuilder = new Fix.Builder(violation);
        for (Cell cell : cells) {
            if (cell.hasColumnName("city")) {
                Fix fix = fixBuilder.left(cell).right("amsterdam").build();
                result.add(fix);
            }
        }

         return result;
    }
}
