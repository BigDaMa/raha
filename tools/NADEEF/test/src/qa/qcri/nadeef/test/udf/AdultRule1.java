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
 * Test rule.
 */
public class AdultRule1 extends SingleTupleRule {
    /**
     * Detect rule with one tuple.
     *
     * @param tuple input tuple.
     * @return Violation set.
     */
    @Override
    public Collection<Violation> detect(Tuple tuple) {
        List<Violation> result = Lists.newArrayList();

        String martialStatus = (String)tuple.get("martialstatus");
        String relationship = (String)tuple.get("relationship");
        if (
            martialStatus.equalsIgnoreCase("Divorced") ||
            martialStatus.equalsIgnoreCase("Never-married")
        ) {
            if (
                relationship.equalsIgnoreCase("husband") ||
                relationship.equalsIgnoreCase("wife")
            ) {
                Violation violation = new Violation(getRuleName());
                violation.addCell(tuple.getCell("martialstatus"));
                violation.addCell(tuple.getCell("relationship"));
                result.add(violation);
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
        List<Cell> violatedCells = Lists.newArrayList(violation.getCells());
        Fix.Builder fixBuilder = new Fix.Builder(violation);
        for (Cell cell : violatedCells) {
            if (cell.hasColumnName("relationship")) {
                Fix fix = fixBuilder.left(cell).right("Unmarried").build();
                result.add(fix);
            }
        }

        return result;
    }
}
